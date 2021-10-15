import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import context
from mindspore.common import Parameter
#  from dataset import create_yolo_dataset
from train import prepare_dataset
import participant_model
from participant_model import tobox
from evaluate.evaluate import evaluate


context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)


def get_model_net_with_weights(ckpt_file='./model.ckpt'):
    net = participant_model.Net()
    param_dict = ms.load_checkpoint(ckpt_file)
    if context.get_context("device_target") == "GPU":
        print('cast back to float32')
        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
    ms.load_param_into_net(net, param_dict)
    #  net.set_train(False)  # for yolov3
    return net


def run_eval(opt):
    net = get_model_net_with_weights()

    meta_df = pd.read_csv('metadata.csv')
    labels_df = pd.read_csv('labels.csv')

    num_class = {0: 'SCC', 1: 'AC', 2: 'SCLC', 3: 'NSCLC'}
    rename_mapping = {v.lower(): f'p{k}' for k, v in num_class.items()}
    labels_df = labels_df.rename(columns=rename_mapping)
    labels_df['probability_sum'] = labels_df.apply(
        lambda row: sum([row[f'p{i}']for i in range(len(num_class))]),
        axis=1
    )

    prediction_df = pd.DataFrame(columns=[
        'image_id', 'xmin', 'ymin', 'xmax', 'ymax',
        'p0', 'p1', 'p2', 'p3', 'probability_sum'
    ])
    for img_iter, row in meta_df.iterrows():
        image_id = row['image_id']
        image_path = os.path.join('dataset', 'images', row['image_id'] + '.bmp')
        image = np.asarray(Image.open(image_path), dtype=np.uint8).transpose((2, 0, 1))
        image_shape = (row['height'], row['width'])

        print(image_id)

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
        #  xai_results.append(xai_result)

        for bbox in result:
            p = {'image_id': image_id}
            p.update({k: bbox[i] for i, k in enumerate(['xmin', 'ymin', 'xmax', 'ymax'])})
            p.update({f'p{i}': bbox[4+i] for i in range(4)})
            prediction_df = prediction_df.append(p, ignore_index=True)
        draw_prediction(opt, image_id, result, labels_df, num_class)

        if img_iter > 10:
            break

    prediction_df['probability_sum'] = prediction_df.apply(
        lambda row: sum([row[f'p{i}']for i in range(len(num_class))]),
        axis=1
    )
    print(prediction_df)
    froc_score = evaluate(
        prediction_df, labels_df,
        fp_sampling=[1/4, 1/2, 1, 2, 4, 8]
    )
    print('froc = {}'.format(froc_score))


def draw_prediction(opt, image_id, pred, labels_df, num_class):
    opt.draw_dir = os.path.join(opt.output_url, 'draw')
    if not os.path.exists(opt.draw_dir):
        os.makedirs(opt.draw_dir)

    boxes = pred[:, :4]
    box_scores = pred[:, 4:]
    classes = np.argmax(box_scores, axis=1)
    scores = np.max(box_scores, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    image_path = os.path.join('dataset', 'images', image_id + '.bmp')
    f = Image.open(image_path)
    img_np = np.asarray(f, dtype=np.float32)
    ax.imshow(img_np.astype(np.uint8))
    h, w, _ = img_np.shape

    # draw prediction
    for box_index in range(boxes.shape[0]):
        #  ymin = boxes[box_index][0] * w
        #  xmin = boxes[box_index][1] * h
        #  ymax = boxes[box_index][2] * w
        #  xmax = boxes[box_index][3] * h
        xmin = boxes[box_index][0] * w
        ymin = boxes[box_index][1] * h
        xmax = boxes[box_index][2] * w
        ymax = boxes[box_index][3] * h
        ax.add_patch(plt.Rectangle(
            (xmin,ymin), (xmax-xmin), (ymax-ymin),
            fill=False, edgecolor='red', linewidth=1
        ))
        ax.text(
            xmin, ymin,
            s=f'{num_class[classes[box_index]]}: {scores[box_index]}',
            style='italic', c='red', bbox={'alpha': 0.5}
        )

    # draw groundtruth
    annotation = labels_df.loc[labels_df['image_id'] == image_id]
    for it, row in annotation.iterrows():
        xmin = row['xmin'] * w
        ymin = row['ymin'] * h
        xmax = row['xmax'] * w
        ymax = row['ymax'] * h
        ax.add_patch(plt.Rectangle(
            (xmin,ymin), (xmax-xmin), (ymax-ymin),
            fill=False, edgecolor='blue', linewidth=1
        ))
        labels = [v for k, v in num_class.items() if row[f'p{k}'] == 1]
        labels = ','.join(labels)
        ax.text(
            xmin, ymin, s=labels,
            style='italic', c='blue', bbox={'alpha': 0.5}
        )



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

