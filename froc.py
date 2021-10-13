#!/usr/bin/env python
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

predefined_fp = [1/4, 1/2, 1, 2, 4, 8]

def froc(
        tp_prob: np.array,
        fp_prob: np.array,
    
        gt_count: int,
        image_count: int,
    
        predefined_fp: np.array
    ):
    """
    Free-ROC
    
    Args:
        tp_prob:        np.array
        fp_prob:        np.array, 
        gt_count:       int, number of ground truth bounding box
        image_count:    int, number of image samples
        predefined_fp:  np.array, predifined fp/img
    
    Returns:
        score:    float 
    """
    assert gt_count >= len(tp_prob), "There cannot be more true guesses than numbers of ground truth."
    
    join_probs = np.unique(np.concatenate((tp_prob, fp_prob)))  # consider rounding to make this operation faster.
    join_probs = np.around(join_probs, decimals=8)
    
    if len(join_probs) == 0:
        return 0 # no score for not guess anything
    if gt_count == 0:
        assert image_count == 0, 'Ground truth count is 0 but image count is not ??'
        return 0
    
    # accumulate the count, there is another algorithm for this but we ignore first.
    tp_count = np.zeros(len(join_probs))
    fp_count = np.zeros(len(join_probs))
    
    for idx, join_prob in enumerate(join_probs):
        tp_count[idx] = np.count_nonzero(tp_prob >= join_prob)
        fp_count[idx] = np.count_nonzero(fp_prob >= join_prob)
    tp_rate = tp_count / gt_count
    fp_per_img = fp_count / image_count
    tp_rate = np.append(tp_rate, 0)[::-1]
    fp_per_img = np.append(fp_per_img,0)[::-1]
    
    predefined_fp = np.sort(predefined_fp)
    interpolation_function = interpolate.interp1d(fp_per_img, tp_rate, fill_value=(0, max(tp_rate)), bounds_error=False)
    
    xnew = np.arange(0, 10, 0.01)
    ynew = interpolation_function(xnew)   # use interpolation function returned by `interp1d`
    plt.plot(xnew, ynew, '-')
    
    axes = plt.gca()
    axes.set_xlim([0, 10])
    
    plt.legend(['NSCLC', 'SCLC', 'SCC', 'AC'])
    plt.xlabel('False Positive per Image')
    plt.ylabel('Recall')
    
    plt.savefig(f'./froc.png')
    
    return sum(
        interpolation_function(predefined_fp)
    ) / len(predefined_fp)  # get the average of these interpolations,

    
    
