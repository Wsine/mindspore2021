import os
import argparse
import ast
from easydict import EasyDict as edict
import shutil
from pathlib import Path

import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer
from mindspore.nn.dynamic_lr import warmup_lr

from yolov3 import yolov3_resnet18, YoloWithLossCell, TrainingWrapper
from dataset import create_yolo_dataset, data_to_mindrecord_byte_image
from config import ConfigYOLOV3ResNet18


def get_lr(learning_rate, start_step, global_step, decay_step, decay_rate, steps=False):
    """Set learning rate."""
    lr_each_step = []
    for i in range(global_step):
        if steps:
            lr_each_step.append(learning_rate * (decay_rate ** (i // decay_step)))
        else:
            lr_each_step.append(learning_rate * (decay_rate ** (i / decay_step)))
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    lr_each_step = lr_each_step[start_step:]
    return lr_each_step


def init_net_param(network, init_value='ones'):
    """Init the parameters in network."""
    params = network.trainable_params()
    for p in params:
        if isinstance(p.data, Tensor) and 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            p.set_data(initializer(init_value, p.data.shape, p.data.dtype))


def run_train(opt):
    if opt.is_modelarts is True:
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=opt.device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=opt.device_id)
    #  context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=opt.device_id)

    rank = 0
    device_num = 1
    if opt.distribute:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()

    loss_scale = float(opt.loss_scale)

    # When create MindDataset, using the fitst mindrecord file, such as yolo.mindrecord0.
    dataset = create_yolo_dataset(opt.mindrecord_file,
                                  batch_size=opt.batch_size, device_num=device_num, rank=rank)
    dataset_size = dataset.get_dataset_size()
    print(f"Dataset Created with num of batches: {dataset_size}")

    net = yolov3_resnet18(ConfigYOLOV3ResNet18())
    net = YoloWithLossCell(net, ConfigYOLOV3ResNet18())
    init_net_param(net, "XavierUniform")

    # checkpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * opt.save_checkpoint_epochs,
                                  keep_checkpoint_max=opt.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="yolov3", directory=opt.ckpt_dir, config=ckpt_config)

    total_epoch_size = 60
    if opt.distribute:
        total_epoch_size = 160

    lr = Tensor(get_lr(learning_rate=opt.lr, start_step=0,
                       global_step=total_epoch_size * dataset_size,
                       decay_step=1000, decay_rate=0.95, steps=True))
    optim = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), lr, loss_scale=loss_scale)
    net = TrainingWrapper(net, optim, loss_scale)

    callback = [LossMonitor(1*dataset_size), ckpoint_cb]

    model = Model(net)
    dataset_sink_mode = opt.dataset_sink_mode
    print("============ Start Training ============")
    print("The first epoch will be slower because of the graph compilation.")
    model.train(opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)


def prepare_dataset(opt):
    opt.ckpt_dir = os.path.join(opt.output_url, 'ckpt')
    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)

    data_path = './dataset/'
    label_path = './labels.csv'
    metadata_path = './metadata.csv'
    if opt.is_modelarts is True:
        import moxing as mox
        data_url = 'obs://msc21-dataset2/dataset/'
        label_url = 'obs://msc21-dataset2/dataset/labels.csv'
        metadata_url = 'obs://msc21-dataset2/dataset/metadata.csv'
        if not os.path.exists(data_path):
            mox.file.copy_parallel(src_url=data_url, dst_url=data_path)
        if not os.path.exists(label_path):
            mox.file.copy_parallel(src_url=label_url, dst_url=label_path)
        if not os.path.exists(metadata_path):
            mox.file.copy_parallel(src_url=metadata_url, dst_url=metadata_path)

    mindrecord_dir_train = os.path.join(data_path, 'mindrecord')

    print("Start creating dataset!")
    # It will generate mindrecord file in args_opt.mindrecord_dir,and the file name is yolo.mindrecord.
    prefix = "yolo.mindrecord"
    opt.mindrecord_file = os.path.join(mindrecord_dir_train, 'train', prefix)
    if os.path.exists(mindrecord_dir_train):
        print('The mindrecord file had exists!')
    else:
        image_dir = os.path.join(data_path, 'images')
        if not os.path.exists(mindrecord_dir_train):
            os.makedirs(mindrecord_dir_train)
        print("Start Creating Mindrecord!")

        mindrecord_path = Path(mindrecord_dir_train)
        mindrecord_train_path = mindrecord_path / 'train'
        mindrecord_test_path = mindrecord_path / 'test'

        mindrecord_path.mkdir(parents=True, exist_ok=True)
        mindrecord_train_path.mkdir(parents=True, exist_ok=True)
        mindrecord_test_path.mkdir(parents=True, exist_ok=True)

        data_to_mindrecord_byte_image(
            Path(image_dir), Path(mindrecord_dir_train), prefix, 1,
            Path(label_path), Path(metadata_path), train_test_split=0.95
        )
        print("Mindrecord Created at {}".format(mindrecord_dir_train))


if __name__ == '__main__':
    # The parameter: "args_opt.output_url" refers to the local location of the job training instance.
    # All the output can save to here, and it will upload all of the directory to the specified OBS location after the training.
    parser = argparse.ArgumentParser(description='Yolo Training')
    parser.add_argument('--output_url', required=True, default=None, help='Location of training outputs.')
    parser.add_argument('--is_modelarts', type=bool, default=False)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--loss_scale', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch_size', type=int, default=50)
    parser.add_argument('--save_checkpoint_epochs', type=int, default=1)
    parser.add_argument('--keep_checkpoint_max', type=int, default=10)
    parser.add_argument('--dataset_sink_mode', type=bool, default=True)
    parser.add_argument('--distribute', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001)

    opt = parser.parse_args()
    print(opt)

    prepare_dataset(opt)
    run_train(opt)

