import os
import numpy as np
import cv2
from PIL import Image
import slidingwindow as sw
import scipy as sp

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore.explainer.explanation import Occlusion

# yolov3
#  from yolov3.yolov3 import yolov3_resnet18, YoloWithEval
#  from yolov3.config import ConfigYOLOV3ResNet18
# fastrcnn
from fasterrcnn.faster_rcnn_resnet import Faster_Rcnn_Resnet
from fasterrcnn.config import ConfigFastRCNN
from fasterrcnn.dataset import rescale_column_test, resize_column_test, imnormalize_column, transpose_column
from mindspore.explainer.explanation import Occlusion
from mindspore.explainer.explanation import Gradient
from mindspore.explainer.explanation import Deconvolution
from mindspore.explainer.explanation import GuidedBackprop


class BambooConvolution(ms.nn.Cell):
    def __init__(self, input_channels=3):
        super(BambooConvolution, self).__init__()
        self.conv2d_1 = ms.nn.Conv2d(input_channels, 16, 3, pad_mode='same')
        self.conv2d_2 = ms.nn.Conv2d(16, 32, 3, pad_mode='same')
        self.conv2d_3 = ms.nn.Conv2d(32, 32, 3, pad_mode='same')

        self.dense_1 = ms.nn.Dense(2048, 64)
        self.dense_2 = ms.nn.Dense(64, 2)

        self.activation = ms.nn.ReLU()
        self.classifier = ms.nn.Sigmoid()
        self.pooling = ms.nn.MaxPool2d(2, 2)
        self.dropout = ms.nn.Dropout(0.9)
        self.flatten = ms.nn.Flatten()
        self.add = ms.ops.Add()
        self.concat = ms.ops.Concat(1)

    def construct(
        self, x
    ):
        x = self.conv2d_1(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.dropout(x)


        x = self.conv2d_2(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.dropout(x)

        """
        x2 = self.pooling(x2)

        x = self.concat((x, x2))
        x = self.activation(x)
        """

        x = self.conv2d_3(x)
        x = self.activation(x)
        x = self.pooling(x)
        x = self.dropout(x)
        """
        ms.ops.Print()(
        ms.ops.Shape()(x)
        )
        """
        x = self.flatten(x)

        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.dense_2(x)

        return x


image_size, slide_size = (64, 0.5)
state = {}


class FusionNet(ms.nn.Cell):
    def __init__(self, rcnn_net, bamboo_net):
        super(FusionNet, self).__init__()
        self.rcnn_net = rcnn_net
        self.bamboo_net = bamboo_net

    def construct(self, img_data, img_metas, patches):
        output1 = self.rcnn_net(img_data, img_metas)
        output2 = self.bamboo_net(patches)
        return output1, output2


def np_softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


#  cfg = ConfigYOLOV3ResNet18()
cfg = ConfigFastRCNN()


def Net():
    #  net = yolov3_resnet18(cfg)
    #  eval_net = YoloWithEval(net, cfg)
    net1 = Faster_Rcnn_Resnet(cfg)
    if context.get_context('device_target') == 'Ascend':
        net1.to_float(mstype.float16)
    net2 = BambooConvolution()
    eval_net = FusionNet(net1, net2)
    eval_net.set_train(False)
    return eval_net


def _infer_data_yolov3(img_data, input_shape):
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


def _infer_data_fastrcnn(image):
    image_bgr = image.copy()
    image_bgr[:, :, 0] = image[:, :, 2]
    image_bgr[:, :, 1] = image[:, :, 1]
    image_bgr[:, :, 2] = image[:, :, 0]
    image_shape = image_bgr.shape[:2]
    box_shape = 1
    fake_box = np.asarray([[1, 2, 1, 2]])
    fake_label = np.asarray([0])
    fake_iscrowd = np.asarray([0])

    pad_max_number = 128
    gt_box_new = np.pad(fake_box, ((0, pad_max_number - box_shape), (0, 0)), mode="constant", constant_values=0)
    gt_label_new = np.pad(fake_label, ((0, pad_max_number - box_shape)), mode="constant", constant_values=-1)
    gt_iscrowd_new = np.pad(fake_iscrowd, ((0, pad_max_number - box_shape)), mode="constant", constant_values=1)
    gt_iscrowd_new_revert = (~(gt_iscrowd_new.astype(np.bool))).astype(np.int32)


    image_shape = image_shape[:2]
    input_data = image_bgr, image_shape, gt_box_new, gt_label_new, gt_iscrowd_new_revert

    if cfg.keep_ratio:
        input_data = rescale_column_test(*input_data, config=cfg)
    else:
        input_data = resize_column_test(*input_data, config=cfg)
    input_data = imnormalize_column(*input_data)

    output_data = transpose_column(*input_data)
    return output_data


def pre_process(iid, image):
    """Data augmentation function."""
    print('pre_process:', iid)

    #  image_size = cfg.img_shape
    #  image = image.transpose((1, 2 ,0))
    #  if not isinstance(image, Image.Image):
    #      image = Image.fromarray(image)
    #  image, image_size = _infer_data_yolov3(image, image_size)
    #  return {
    #      'x': Tensor(image),
    #      'image_shape': Tensor(image_size)
    #  }

    boo_image = image.copy()
    image_id = iid

    image = image.transpose((1, 2 ,0))
    img_data, img_shape, _, _, _ = _infer_data_fastrcnn(image)
    img_data = np.expand_dims(img_data, 0)
    img_shape = np.stack([img_shape])

    boo_image = boo_image.transpose((1, 2 ,0))
    boo_image = boo_image / 255
    boo_image = boo_image.astype(np.float32)
    patches = []
    windows = sw.generate(boo_image, sw.DimOrder.HeightWidthChannel, image_size, slide_size)
    for i,window in enumerate(windows):
        _img = boo_image[window.indices()]
        patches.append(_img)
    patches = np.array(patches)
    #check total images, how many images are tiled at height direction, and width direction
    n_total = len(windows)
    _x = 0
    for i, window in enumerate(windows):
        if _x != window.x:
            n_x = i
            break
        _x = window.x
    state[image_id] = (n_total, n_x, boo_image.shape[0], boo_image.shape[1])
    patches = Tensor(np.transpose(patches, (0,3,1,2)), ms.float32)

    return {
        'img_data': Tensor(img_data),
        'img_metas': Tensor(img_shape),
        'patches': patches
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
        #  classes = np.zeros((len(nms_index), 4), dtype=np.int32)
        #  if c == 0:
        #      classes[:, [0,3]] = 1
        #  elif c == 1:
        #      classes[:, [1,3]] = 1
        #  else:
        #      classes[:, c] = 1
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes = np.concatenate(boxes_, axis=0)
    classes = np.concatenate(classes_, axis=0)
    scores = np.concatenate(scores_, axis=0)

    return boxes, classes, scores


def post_process(iid, prediction):
    print('post_process:', iid)
    # for yolov3
    #  pred_boxes, pred_scores, image_shape = prediction
    #  pred_boxes = pred_boxes.asnumpy()[0]
    #  pred_scores = pred_scores.asnumpy()[0]
    #  pred_boxes[:, [0,1]] = pred_boxes[:, [1,0]]
    #  pred_boxes[:, [2,3]] = pred_boxes[:, [3,2]]
    #  boxes, classes, scores = tobox(pred_boxes, pred_scores)

    output1, output2 = prediction

    # for fastrcnn
    (all_bbox, all_label, all_mask), img_metas = output1

    max_num = 48
    all_bbox_squee = np.squeeze(all_bbox.asnumpy()[0, :, :])
    all_label_squee = np.squeeze(all_label.asnumpy()[0, :, :])
    all_mask_squee = np.squeeze(all_mask.asnumpy()[0, :, :])

    all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
    all_labels_tmp_mask = all_label_squee[all_mask_squee]

    if all_bboxes_tmp_mask.shape[0] > max_num:
        inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
        inds = inds[:max_num]
        all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
        all_labels_tmp_mask = all_labels_tmp_mask[inds]

    # num_classes (int): class number, including background class
    if all_bboxes_tmp_mask.shape[0] == 0:
        result = [np.zeros((0, 5), dtype=np.float32) for _ in range(cfg.num_classes - 1)]
    else:
        result = [all_bboxes_tmp_mask[all_labels_tmp_mask == i, :] for i in range(cfg.num_classes - 1)]
        #  result = []
        #  for i in range(cfg.num_classes - 1):
        #      if i < 2:
        #          result.append(np.zeros((0, 5), dtype=np.float32))
        #      else:
        #          result.append(all_bboxes_tmp_mask[all_labels_tmp_mask == i, :])
    print('------------------result-------------')
    #  print(result)

    boxes = []
    classes = []
    for i, res in enumerate(result):
        boxes.append(res[:, :4])
        #  clss = np.zeros((res.shape[0], cfg.num_classes - 1), dtype=np.int32)
        clss = np.zeros((res.shape[0], cfg.num_classes - 1))
        #  clss[:, i] = 1
        clss[:, i] = res[:, 4]
        classes.append(clss)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    image_shape = img_metas[0][:2]

    # softmax
    #  classes = np_softmax(classes)

    h, w = image_shape.asnumpy()
    boxes = boxes / np.array([w, h, w, h])
    boxes = boxes.clip(0, 1)
    # pred_classes[:,[0,1,2,3]] = pred_classes[:,[0,1,3,2]]
    classes = classes.clip(0, 1)

    result1 = np.concatenate([boxes, classes], axis=1)
    #  result = result[result[:, [4,5,6,7]].sum(axis=1) > 1e-5]


    image_id = iid
    n_total, n_x, h, w = state[image_id]
    output2 = output2.asnumpy()
    output2 = sp.special.softmax(output2, axis=1)
    output2 = np.reshape(output2[:, 1],(n_total // n_x, n_x))

    result2 = []
    for j, row_probability in enumerate(output2):
        for i, cell_probability in enumerate(row_probability):
            if cell_probability > 0.4:
                result2.append([
                    j * (image_size * slide_size) / w,
                    i * (image_size * slide_size) / h,
                    (j+1) * (image_size * slide_size) / w,
                    (i+1) * (image_size * slide_size) / h,
                    cell_probability,
                    cell_probability,
                    cell_probability,
                    cell_probability
                ])
    result2 = np.array(result2) if len(result2) > 0 else np.zeros((0, 8))

    result = np.concatenate([result1, result2])
    return np.array(result)



#  def saliency_map(net, image_id, image, prediction):
#      context.set_context(mode=context.PYNATIVE_MODE)
#      image = image.transpose((1, 2, 0))
#      image = image / 255
#      image = image.astype(np.float32)
#
#      # saliency_op = Occlusion(net, activation_fn=ms.nn.Softmax()) # score=0.7776
#      # saliency_op3 = Gradient(net) # score=0.8451
#      saliency_op2 = GuidedBackprop(net.bamboo_net) # score=0.8704
#      saliency_op1 = Deconvolution(net.bamboo_net) # score=0.8858
#
#      def process(data, batch):
#
#          transformed  = [window.apply(data) for window in batch]  # (1, 64, 64, 3)
#
#          img = np.array(transformed).transpose((0, 3, 1, 2))  # (1, 3, 64, 64)
#
#          img_tensor = ms.Tensor(img, ms.float32)
#
#          saliency1 = saliency_op1(img_tensor, 1) # (1, 1, 64, 64)
#          saliency2 = saliency_op2(img_tensor, 1) # (1, 1, 64, 64)
#          # saliency3 = saliency_op3(img_tensor, 1) # (1, 1, 64, 64)
#
#          return (saliency1.asnumpy()[0][0] + saliency2.asnumpy()[0][0]) / 2  # (64, 64)
#
#      result = sw.mergeWindows(image, sw.DimOrder.HeightWidthChannel, image_size, slide_size, 1, process)
#
#      return np.sum(result, axis=2) / 64
#
