import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import mindspore as ms
from dataset import create_yolo_dataset
from train import prepare_dataset
import participant_model
from participant_model import tobox


def get_model_net_with_weights(ckpt_file='./yolov3.ckpt'):
    net = participant_model.Net()
    param_dict = ms.load_checkpoint(ckpt_file)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)
    return net


def run_eval(opt):
    net = get_model_net_with_weights()

    dataset = create_yolo_dataset(
        opt.mindrecord_file, batch_size=opt.batch_size,
        device_num=1, rank=0, is_training=False
    )
    dataset_size = dataset.get_dataset_size()
    print(f"Dataset Created with num of batches: {dataset_size}")

    results = []
    xai_results = []
    num_class = {0:'SCC', 1: 'AC', 2:'SCLC', 3: 'NSCLC'}
    for data in dataset.create_dict_iterator(output_numpy=True):
        image_file = data['file']  # type: ignore
        image_id = image_file.tobytes().decode('ascii')
        print(image_id)
        image = data['image']  # type: ignore
        image_shape = data['image_shape']  # type: ignore
        if hasattr(participant_model, 'pre_process'):
            pre_processed_data = participant_model.pre_process(
                image_id, image
            )
        else:
            pre_processed_data = (ms.Tensor(image), ms.Tensor(image_shape))

        if type(pre_processed_data) == dict:
            prediction = net(**pre_processed_data)
        else:
            prediction = net(*pre_processed_data)

        if hasattr(participant_model, 'post_process'):
            result = participant_model.post_process(
                image_id, prediction
            )
        else:
            result = prediction

        #  if hasattr(participant_model, 'saliency_map'):
        #      xai_result = participant_model.saliency_map(
        #          net, image_id, image, prediction
        #      )
        #  else:
        #      xai_result = None

        draw_prediction(opt, image_id, image, result, num_class)

        results.append(result)
        #  xai_results.append(xai_result)


def draw_prediction(opt, image_id, img_np, pred, num_class):
    opt.draw_dir = os.path.join(opt.output_url, 'draw')
    if not os.path.exists(opt.draw_dir):
        os.makedirs(opt.draw_dir)

    boxes = pred[:, :4]
    box_scores = pred[:, 4:]
    #  classes = np.max(box_scores)
    #  boxes = pred[0].asnumpy()[batch_idx]
    #  box_scores = pred[1].asnumpy()[batch_idx]
    #  boxes, classes, scores = tobox(boxes, box_scores)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    image_path = os.path.join('dataset', 'images', image_id)
    f = Image.open(image_path)
    img_np = np.asarray(f, dtype=np.float32)
    ax.imshow(img_np.astype(np.uint8))

    # draw prediction
    for box_index in range(boxes.shape[0]):
        ymin=boxes[box_index][0]
        xmin=boxes[box_index][1]
        ymax=boxes[box_index][2]
        xmax=boxes[box_index][3]
        ax.add_patch(plt.Rectangle(
            (xmin,ymin), (xmax-xmin), (ymax-ymin), fill=False, edgecolor='red', linewidth=1
        ))
        #  ax.text(xmin, ymin, s=str(num_class[classes[box_index]]) + str(scores[box_index]),
        #          style='italic', c ='red', bbox={'alpha': 0.5})
        ax.text(xmin, ymin, s=str(box_scores[box_index]),
                style='italic', c ='red', bbox={'alpha': 0.5})

    save_file = os.path.join(opt.draw_dir, image_id.rstrip('.bmp') + '.png')
    print(save_file)
    plt.savefig(save_file)


if __name__ == '__main__':
    # The parameter: "args_opt.output_url" refers to the local location of the job training instance.
    # All the output can save to here, and it will upload all of the directory to the specified OBS location after the training.
    parser = argparse.ArgumentParser(description='Yolo Training')
    parser.add_argument('--output_url', required=True, default=None, help='Location of training outputs.')
    parser.add_argument('--is_modelarts', type=bool, default=False)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)

    opt = parser.parse_args()
    print(opt)

    prepare_dataset(opt)
    run_eval(opt)

