#!/usr/bin/env python
from evaluate.bbox import Bbox
import numpy as np;

from typing import List

bbox_precision_threshold = 0.5

def tp_fp_from_bboxes(
        ground_truth_bboxes: List[Bbox],
        prediction_bboxes: List[Bbox],
        label: int
    ):
    """
    iop: intersection over prediciton
    if prediction bbox and ground truth bbox have iop > 0.5, then it is true positive
    otherwise its false positive

    but, each prediction bbox can have multiple ground truth bboxes, so the highest iop will be used.
    if there are multiple iop that are the same.
    any random ground truth bboxes will be choosen, but we will try to choose the one that is not taken.

    """

    gt_boxes = ground_truth_bboxes
    pt_boxes = prediction_bboxes

    fps = []
    tps = []
    gt_useds = np.zeros(len(gt_boxes))

    for pt_box in pt_boxes:

        max_gt_itsec = 0
        max_gt_itsec_idx = -1

        pt_box_area = pt_box.area()
        if pt_box_area == 0:
            pt_box_area = 1e-6
        for idx, gt_box in enumerate(gt_boxes):
            pd_gt_iop = pt_box.intersect_area(gt_box) / pt_box_area
            if pd_gt_iop > bbox_precision_threshold:
                if pd_gt_iop > max_gt_itsec or (gt_useds[idx] == 0 and pd_gt_iop == max_gt_itsec):
                    max_gt_itsec = pd_gt_iop
                    max_gt_itsec_idx = idx

        # could not find any bbox
        if max_gt_itsec_idx == -1:
            fps.append(pt_box[f'p{label}'])
        else:
            # the maximum intersection ground truth bbox, check if the probability is higher than the maximum
            if gt_useds[max_gt_itsec_idx] < pt_box[f'p{label}']:
                gt_useds[max_gt_itsec_idx] = pt_box[f'p{label}']

    tps = gt_useds[gt_useds > 0]  # true positive probabilities
    fps = np.array(fps) # false positive probabilities

    return tps, fps

