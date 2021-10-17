# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""FasterRcnn dataset"""
from __future__ import division

import os
from pathlib import Path
import numpy as np
from numpy import random
import pandas as pd
import slidingwindow as sw

import cv2
import mindspore as ms
import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C
from mindspore.mindrecord import FileWriter

def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


# https://stackoverflow.com/questions/57474814/python-balancing-items-in-a-list-numpy-array
def balance_select(classes):
    # classes is from 0 to n - 1
    c = np.bincount(classes - 1)
    n = c.min()
    # Accumulated counts for each class shifted one position
    cs = np.roll(np.cumsum(c), 1)
    cs[0] = 0
    # Compute appearance index for each class
    i = np.arange(len(classes)) - cs[classes - 1]
    # Mask excessive appearances
    m = i < n
    return m


def preprocess_fn(image, box, file, is_training, config):
    """Preprocess function for dataset."""
    h, w = image.shape[:2]
    #  image = image / 255
    #  print('image shape', image.shape)
    windows = sw.generate(image, sw.DimOrder.HeightWidthChannel, config.image_size, config.overlap_percent)
    patches = []
    patches_coord = []
    for win in windows:
        patches.append(image[win.indices()])
        wx, wy, ww, wh = win.getRect()
        patches_coord.append((wx, wy, wx+ww, wy+wh))
    patches = np.array(patches)
    patches_coord = np.array(patches_coord)
    box_coord = box[:, :4]
    iou = bbox_overlaps(patches_coord, box_coord)
    has_object = (iou > config.iou_ratio).sum(axis=1).astype(bool).astype(int)
    object_class = np.where(has_object, np.argmax(iou, axis=1), np.zeros_like(has_object))
    #  object_class = np.eye(config.num_classes)[object_class]

    ho = has_object.nonzero()[0]
    nho = (~(has_object.astype(bool))).nonzero()[0]
    ind = np.concatenate([ho, nho])[:config.win_batch_size]
    np.random.shuffle(ind)
    patches = patches[ind]
    has_object = has_object[ind]
    object_class = object_class[ind]

    patches = patches.transpose((0, 3, 1, 2)).astype(np.float32)
    #  print('patches shape', patches.shape)

    #  image_shape = np.array([h, w], dtype=np.int32)
    #  return patches, image_shape, has_object, object_class

    return patches, has_object, object_class


def filter_valid_data(label_file):
    """Filter valid image file, which both in image_dir and anno_path."""

    label_df = pd.read_csv(label_file)
    label_dict = dict(list(label_df.groupby('image_id')))

    label_group_by = label_df.groupby('image_id')
    label_group_by = label_group_by.apply(lambda x: x[['xmin','ymin','xmax','ymax','scc','ac','sclc','nsclc']].to_numpy())
    return label_group_by.keys(), label_group_by


def data_to_mindrecord_byte_image(
    image_dir: Path, mindrecord_dir: Path, prefix, file_num,
    label_file,
    metadata_file,
    train_test_split=0.8
):
    """Create MindRecord file by image_dir and anno_path."""
    mindrecord_train_path = str(mindrecord_dir / 'train' / prefix)
    mindrecord_test_path = str(mindrecord_dir / 'test' / prefix)

    writer_train = FileWriter(mindrecord_train_path, file_num)
    writer_test = FileWriter(mindrecord_test_path, file_num)

    image_files, image_anno_dict = filter_valid_data(label_file)
    num_classes = 4

    metadata_df = pd.read_csv(metadata_file)

    yolo_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 4 + num_classes]},
        "file": {"type": "string"},
    }
    writer_train.add_schema(yolo_json, "yolo_json")
    writer_test.add_schema(yolo_json, "yolo_json")

    print('number of groups of images:', len(image_files))
    split_at = len(image_files)*train_test_split
    print('number of training groups of images:', split_at)
    print('number of test groups of images:', len(image_files) - split_at)

    for idx, image_name in enumerate(image_files):
        image_path = os.path.join(image_dir, image_name + '.bmp')
        with open(image_path, 'rb') as f:
            img = f.read()
        image_metadata = metadata_df[metadata_df.image_id == image_name].iloc[0]
        w = image_metadata.width;
        h = image_metadata.height;
        image_anno_dict[image_name][:,[0,2]] *= w
        image_anno_dict[image_name][:,[1,3]] *= h
        annos = np.array(image_anno_dict[image_name],dtype=np.int32)

        row = {"image": img, "annotation": annos, "file": image_name+'.bmp'}
        if idx < split_at:
            writer_train.write_raw_data([row])
        else:
            writer_test.write_raw_data([row])

    writer_train.commit()
    writer_test.commit()


def create_resnet_dataset(config, mindrecord_file, batch_size=2, device_num=1, rank_id=0, is_training=True, num_parallel_workers=8, python_multiprocessing=False):
    """Create ResNet dataset with MindDataset."""
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(8)
    ds = de.MindDataset(mindrecord_file, columns_list=["image", "annotation", "file"], num_shards=device_num, shard_id=rank_id, num_parallel_workers=4, shuffle=is_training)
    decode = C.Decode()
    ds = ds.map(input_columns=["image"], operations=decode)
    compose_map_func = (lambda image, annotation, file: preprocess_fn(image, annotation, file, is_training, config=config))

    if is_training:
        ds = ds.map(input_columns=["image", "annotation", "file"],
                    output_columns=["images", "has_object", "object_class"],
                    column_order=["images", "has_object", "object_class"],
                    operations=compose_map_func, python_multiprocessing=python_multiprocessing,
                    num_parallel_workers=num_parallel_workers)
        #  ds = ds.batch(batch_size, drop_remainder=True)
    else:
        ds = ds.map(input_columns=["image", "annotation", "file"],
                    output_columns=["images", "has_object", "object_class"],
                    column_order=["images", "has_object", "object_class"],
                    operations=compose_map_func,
                    num_parallel_workers=num_parallel_workers)
        #  ds = ds.batch(batch_size, drop_remainder=True)
    return ds
