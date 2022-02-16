import numpy as np
from sed_scores_eval.utils.array_ops import get_first_index_where


def get_curve_idx_for_threshold(scores, threshold):
    """get that index od a curve that corresponds to a given threshold
    
    Args:
        scores: 
        threshold: 

    Returns:

    """
    if not np.isscalar(threshold):
        raise ValueError(
            f'threshold_of_interest must be scalar but {threshold}'
            f' was given'
        )
    sort_idx = np.argsort(scores)
    idx = get_first_index_where(scores[sort_idx], 'gt', threshold)
    if idx == len(sort_idx):
        assert scores[sort_idx[-1]] == np.inf, scores[sort_idx[-1]]
        idx = -1
    return sort_idx[idx]
