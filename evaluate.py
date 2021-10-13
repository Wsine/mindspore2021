#!/usr/bin/env python
from typing import List

import numpy as np
import pandas as pd

from bbox import read_participants_input
from evaluate_saliency import evaluate_saliency
from evaluate_bbox import tp_fp_from_bboxes
from froc import froc

# evaluate the FROC for prediction and ground truth bounding boxes
def evaluate(
    prediction_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    fp_sampling: List[float]
):
    """
    Perform FROC calculation on the data.

    prediction_df and ground_truth_df have columns:
        <image_id, xmin, ymin, xmax, ymax, p0, p1, p2, p3, probability_sum>

    fp_sampling is the false positive per image values for sampling.    

    """

    prediction_df['image_id'] = prediction_df['image_id'].astype(str)
    ground_truth_df['image_id'] = ground_truth_df['image_id'].astype(str)

    if len(prediction_df) > 1000000:
        raise ValueError(
            'Your output have too many columns, only the first 100000 highest probabilties '
            'predictions(sum of p0 to p3) will be accepted.')
    prediction_df = prediction_df.sort_values(['probability_sum'])
    prediction_df = prediction_df.iloc[:1000000]

    if (
        ~(
            (0 <= prediction_df[['xmin', 'ymin', 'xmax', 'ymax', 'p0', 'p1', 'p2', 'p3']]) &
            (prediction_df[['xmin', 'ymin', 'xmax', 'ymax', 'p0', 'p1', 'p2', 'p3']] <= 1)
        )
    ).any().any():
        raise ValueError(
            f'Bounding box values and probabilities is not within expected range of 0 to 1 (inclusive).')

    frocs = 0
    frocs_count = 0  # because if there is no ground truth, we need to ignore the count.
    for label in range(4):  # for each classes

        pred_label_df = prediction_df
        gt_label_df = ground_truth_df[ground_truth_df[f'p{label}'] != 0]

        tpss = [[]] # all true positive probabilities
        fpss = [[]] # all false positive probabilities

        ground_truth_files = gt_label_df['image_id'].unique()
        fpss.append(
            pred_label_df.loc[
                ~(
                    prediction_df['image_id'].isin(ground_truth_files)
                )
            ][f'p{label}']
            .to_numpy()
            # probabilities of predictions for images that are not in the ground truth
        )
        # remaining predictions that have images in the ground truth
        pred_label_df = pred_label_df[
            pred_label_df['image_id'].isin(ground_truth_files)
        ]

        for ground_truth_file in ground_truth_files:
            # filter files
            pred_label_file_df = pred_label_df[
                pred_label_df['image_id'] == ground_truth_file
            ]
            gt_label_file_df = gt_label_df[
                gt_label_df['image_id'] == ground_truth_file
            ]

            # parse as bbox
            prediction_bbox = read_participants_input(pred_label_file_df)
            ground_truth_bbox = read_participants_input(gt_label_file_df)

            # calculate tp, fp
            tps, fps = tp_fp_from_bboxes(
                ground_truth_bboxes=ground_truth_bbox, prediction_bboxes=prediction_bbox,
                label=label
            )

            tpss.append(tps)
            fpss.append(fps)

        tpss = np.concatenate(tpss)
        fpss = np.concatenate(fpss)

        froc_score = froc(
            tpss, fpss,
            len(gt_label_df),
            len(ground_truth_files),
            fp_sampling
        )
        print(f'Evaluating label {label}, score: {froc_score}')
        
        # accumulate the froc score
        if gt_label_df.shape[0] > 0:
            frocs += froc_score
            frocs_count += 1

    # average the froc score
    return frocs / frocs_count
