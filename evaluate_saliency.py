#!/usr/bin/env python
import numpy as np
from sklearn.metrics import roc_auc_score

# Since there is no ground truth saliency this function is for reference only.
def evaluate_saliency(
    prediction_saliency: np.array, # (H,W) of 0 to 1
    ground_truth_saliency: np.array  # (H,W) of 0 or 1
):
    saliency_score = 0 
    
    ground_truth_saliency = ground_truth_saliency >= 0.9

    if (prediction_saliency < 0).sum() > 0 or (prediction_saliency > 1).sum() > 0:
        raise ValueError('Saliency map values is not within 0 and 1 rejecting result.')
    
    saliency_score += roc_auc_score(ground_truth_saliency.flatten(), prediction_saliency.flatten())
    
    return saliency_score
