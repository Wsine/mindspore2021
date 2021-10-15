import os
import argparse
import ast
from easydict import EasyDict as edict
import shutil
from pathlib import Path

import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import context, Tensor
from mindspore.communication.management import init
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer
from mindspore.nn.dynamic_lr import warmup_lr, exponential_decay_lr

# yolov3
#  from yolov3.yolov3 import yolov3_resnet18, YoloWithLossCell, TrainingWrapper
#  from yolov3.config import ConfigYOLOV3ResNet18
#  from yolov3.dataset import create_yolo_dataset, data_to_mindrecord_byte_image
# fastrrcnn
from fasterrcnn.faster_rcnn_resnet import Faster_Rcnn_Resnet
from fasterrcnn.config import ConfigFastRCNN
from fasterrcnn.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet
from fasterrcnn.lr_schedule import dynamic_lr
from fasterrcnn.dataset import create_fasterrcnn_dataset, data_to_mindrecord_byte_image


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
        context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=opt.device_id)
        #  context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=opt.device_id)
    #  context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=opt.device_id)

    rank = 0
    device_num = 1
    if opt.distribute:
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()

    #  loss_scale = float(opt.loss_scale)

    # When create MindDataset, using the fitst mindrecord file, such as yolo.mindrecord0.
    #  dataset = create_yolo_dataset(opt.mindrecord_file,
    #                                batch_size=opt.batch_size, device_num=device_num, rank=rank)
    config = ConfigFastRCNN()
    dataset = create_fasterrcnn_dataset(
        config, opt.mindrecord_file, batch_size=config.batch_size, device_num=device_num, rank_id=rank)
    dataset_size = dataset.get_dataset_size()
    print(f"Dataset Created with num of batches: {dataset_size}")

    #  net = yolov3_resnet18(ConfigYOLOV3ResNet18())
    #  net = YoloWithLossCell(net, ConfigYOLOV3ResNet18())
    #  init_net_param(net, "XavierUniform")

    net = Faster_Rcnn_Resnet(config).set_train()
    param_dict = ms.load_checkpoint(opt.backbone_ckpt_file)
    ms.load_param_into_net(net.backbone, param_dict)
    loss = LossNet()

    #  total_epoch_size = 60
    #  if opt.distribute:
    #      total_epoch_size = 160
    #  lr = Tensor(get_lr(learning_rate=opt.lr, start_step=0,
    #                     global_step=total_epoch_size * dataset_size,
    #                     decay_step=1000, decay_rate=0.95, steps=True))

    #  total_steps = config.epoch_size * dataset_size
    #  turning_step = 100
    #  lr = warmup_lr(config.base_lr, turning_step, 2, 2)
    #  lr += exponential_decay_lr(config.base_lr, 0.9, total_steps-turning_step, 2, 1)
    #  lr = exponential_decay_lr(opt.lr, 0.99, opt.epoch_size, 2, 1)
    #  lr = Tensor(lr).astype(ms.dtype.float32)
    lr = dynamic_lr(config, dataset_size)
    lr = Tensor(lr, ms.common.dtype.float32)

    #  optim = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), lr, loss_scale=loss_scale)
    #  optim = nn.SGD(
    #      filter(lambda x: x.requires_grad, net.get_parameters()), lr,
    #      momentum=0.937, weight_decay=0.0005, loss_scale=loss_scale
    #  )
    optim = nn.SGD(params=net.trainable_params(), learning_rate=lr,
                   momentum=config.momentum, weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    #  net = TrainingWrapper(net, optim, loss_scale)
    net = WithLossCell(net, loss)
    net = TrainOneStepCell(net, optim, sens=config.loss_scale)

    # checkpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * opt.save_checkpoint_epochs,
                                  keep_checkpoint_max=opt.keep_checkpoint_max)
    #  ckpoint_cb = ModelCheckpoint(prefix="yolov3", directory=opt.ckpt_dir, config=ckpt_config)
    ckpoint_cb = ModelCheckpoint(prefix="fastrcnn", directory=opt.ckpt_dir, config=ckpt_config)
    #  callback = [TimeMonitor(data_size=dataset_size), LossMonitor(1*dataset_size), ckpoint_cb]
    callback = [TimeMonitor(data_size=dataset_size), LossMonitor(), ckpoint_cb]

    model = Model(net)
    #  dataset_sink_mode = opt.dataset_sink_mode
    print("============ Start Training ============")
    print("The first epoch will be slower because of the graph compilation.")
    #  model.train(opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)
    model.train(config.epoch_size, dataset, callbacks=callback, dataset_sink_mode=False)


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
    #  parser.add_argument('--loss_scale', type=int, default=1024)
    #  parser.add_argument('--batch_size', type=int, default=32)
    #  parser.add_argument('--epoch_size', type=int, default=50)
    #  parser.add_argument('--lr', type=float, default=0.001)
    #  parser.add_argument('--dataset_sink_mode', type=bool, default=True)
    parser.add_argument('--save_checkpoint_epochs', type=int, default=1)
    parser.add_argument('--keep_checkpoint_max', type=int, default=50)
    parser.add_argument('--distribute', type=bool, default=False)
    parser.add_argument('--backbone_ckpt_file', type=str, default='resnet_backbone.ckpt')

    opt = parser.parse_args()
    print(opt)

    prepare_dataset(opt)
    run_train(opt)

