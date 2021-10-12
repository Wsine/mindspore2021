import os
import numpy as np
import cv2
from PIL import Image

import mindspore as ms
from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore.explainer.explanation import Occlusion

from yolov3 import yolov3_resnet18, YoloWithEval
from config import ConfigYOLOV3ResNet18


cfg = ConfigYOLOV3ResNet18()


def Net():
    net = yolov3_resnet18(cfg)
    eval_net = YoloWithEval(net, cfg)
    return eval_net


def _infer_data(img_data, input_shape):
    w, h = img_data.size
    input_h, input_w = input_shape
    scale = min(float(input_w) / float(w), float(input_h) / float(h))
    nw = int(w * scale)
    nh = int(h * scale)
    img_data = img_data.resize((nw, nh), Image.BICUBIC)

    new_image = np.zeros((input_h, input_w, 3), np.float32)
    new_image.fill(128)
    img_data = np.array(img_data)
    if len(img_data.shape) == 2:
        img_data = np.expand_dims(img_data, axis=-1)
        img_data = np.concatenate([img_data, img_data, img_data], axis=-1)

    dh = int((input_h - nh) / 2)
    dw = int((input_w - nw) / 2)
    new_image[dh:(nh + dh), dw:(nw + dw), :] = img_data
    new_image /= 255.
    new_image = np.transpose(new_image, (2, 0, 1))
    new_image = np.expand_dims(new_image, 0)
    return new_image, np.array([h, w], np.float32)


def disabled_pre_process(iid, image):
    """Data augmentation function."""
    print('pre_process:', iid)
    image = image.transpose((1, 2 ,0))
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image_size = cfg.img_shape
    image, image_size = _infer_data(image, image_size)
    return {
        'x': Tensor(image),
        'image_shape': Tensor(image_size)
    }


def pre_process(iid, image):
    print('pre_process:', iid)
    image_size = cfg.img_shape
    return {
        'x': Tensor(image),
        'image_shape': Tensor(image_size)
    }


def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return keep


def tobox(boxes, box_scores):
    """Calculate precision and recall of predicted bboxes."""
    config = cfg
    num_classes = config.num_classes
    mask = box_scores >= (config.obj_threshold*0.2)
    boxes_ = []
    scores_ = []
    classes_ = []
    max_boxes = config.nms_max_num
    for c in range(num_classes):
        selected = np.reshape(mask[:, c], [-1]).nonzero()[0]
        class_boxes = np.reshape(boxes, [-1, 4])[np.reshape(mask[:, c], [-1])]
        class_box_scores = np.reshape(box_scores[:, c], [-1])[np.reshape(mask[:, c], [-1])]
        nms_index = apply_nms(class_boxes, class_box_scores, config.nms_threshold, max_boxes)
        #nms_index = apply_nms(class_boxes, class_box_scores, 0.5, max_boxes)
        class_boxes = class_boxes[nms_index]
        class_box_scores = class_box_scores[nms_index]
        #  classes = np.ones_like(class_box_scores, 'int32') * c
        classes = box_scores[selected[nms_index]]
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes = np.concatenate(boxes_, axis=0)
    classes = np.concatenate(classes_, axis=0)
    scores = np.concatenate(scores_, axis=0)

    return boxes, classes, scores


def post_process(iid, prediction):
    print('post_process:', iid)
    pred_boxes, pred_scores, image_shape = prediction
    pred_boxes = pred_boxes.asnumpy()[0]
    pred_scores = pred_scores.asnumpy()[0]
    boxes, classes, scores = tobox(pred_boxes, pred_scores)

    h,w = image_shape.asnumpy()

    boxes = boxes / np.array([w, h, w, h])
    boxes = boxes.clip(0, 1)
    # pred_classes[:,[0,1,2,3]] = pred_classes[:,[0,1,3,2]]
    classes = classes.clip(0, 1)

    result = np.concatenate([boxes, classes], axis=1)
    result = result[result[:, [4,5,6,7]].sum(axis=1) > 1e-5]
    return np.array(result)


def saliency_map(
    net,
    image_id,
    image,
    prediction
):

    context.set_context(mode=context.PYNATIVE_MODE)
    temp_prep = pre_process(image_id, image)
    temp_prep2 = {
        'x': ms.Tensor([temp_prep['x']]),
        'image_shape': ms.Tensor([temp_prep['image_shape']]),
    }

    # target = ms.Tensor(np.ones(pre_processed_data.shape[0]), ms.int32)

    occ = Occlusion(net, activation_fn=ms.nn.Softmax())
    saliency = occ(temp_prep2, 1)

    return saliency.asnumpy()[0][0]

