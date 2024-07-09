import numpy as np
from sed_scores_eval.utils.array_ops import get_first_index_where


def staircase_auc(y, x, max_x=None):
    """Area under Curve (AUC) with staircase type interpolation

    Args:
        y (1d np.ndarray): y values
        x (1d np.ndarray): x values
        max_x: maximal x value. If not None curve is interpolated up to this x
            value. Else AUC is computed up to the maximal value in x array.

    Returns:
        auc: AUC value

    >>> staircase_auc(np.array([0.,1.,2.,3.]), np.array([0.,1.,2.,3.]))
    3.0
    >>> staircase_auc(np.array([0.,1.,2.,3.]), np.array([0.,1.,1.1,1.2]))
    0.2999999999999998
    >>> staircase_auc(np.array([0.,1.,2.,3.]), np.array([0.,1.,2.,3.]), max_x=2.5)
    2.0
    >>> staircase_auc(np.array([0.,1.,2.,3.]), np.array([0.,1.,2.,3.]), max_x=10.)
    24.0
    """
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    if max_x is None:
        y = y[:-1]
    else:
        cutoff_idx = get_first_index_where(x, "gt", max_x)
        assert cutoff_idx > 0, cutoff_idx
        x = np.concatenate((x[:cutoff_idx], [max_x]))
        y = y[:cutoff_idx]
    widths = x[1:] - x[:-1]
    areas = y * widths
    auc = sum(areas)
    return auc


def linear_auc(y, x, max_x=None):
    """Area under Curve (AUC) with linear interpolation

    Args:
        y (1d np.ndarray): y values
        x (1d np.ndarray): x values
        max_x: maximal x value. If not None curve is interpolated up to this x
            value. Else AUC is computed up to the maximal value in x array.

    Returns:
        auc: AUC value

    >>> linear_auc(np.array([0.,1.,2.,3.]), np.array([0.,1.,2.,3.]))
    4.5
    >>> linear_auc(np.array([0.,1.,2.,3.]), np.array([0.,1.,1.1,1.2]))
    0.8999999999999998
    >>> linear_auc(np.array([0.,1.,2.,3.]), np.array([0.,1.,2.,3.]), max_x=2.5)
    3.125
    """
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    if max_x is not None:
        cutoff_idx = get_first_index_where(x, "gt", max_x)
        assert cutoff_idx > 0, (cutoff_idx, x[0], max_x)
        assert cutoff_idx < len(x), (len(x), cutoff_idx, x[-1], max_x)
        if x[cutoff_idx] <= x[cutoff_idx-1]:
            y_last = y[cutoff_idx]
        else:
            y_last = y[cutoff_idx-1] + (max_x - x[cutoff_idx-1])/(x[cutoff_idx] - x[cutoff_idx-1])*(y[cutoff_idx]-y[cutoff_idx-1])
        x = np.concatenate((x[:cutoff_idx], [max_x]))
        y = np.concatenate((y[:cutoff_idx], [y_last]))
    y = (y + np.concatenate((y[1:], [y[-1]]))) / 2
    return staircase_auc(y, x)


def get_curve_idx_for_threshold(scores, threshold):
    """get that index of a curve that corresponds to a given threshold

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


def xsort(y, x, *args):
    sort_idx = sorted(np.arange(len(x)).tolist(), key=lambda i: (x[i], y[i]))

    def sort_arg(stat):
        if np.isscalar(stat):
            return stat
        if isinstance(stat, (list, tuple)):
            return [sort_arg(stat_i) for stat_i in stat]
        if isinstance(stat, dict):
            return {key: sort_arg(stat_i) for key, stat_i in stat.items()}
        return stat[sort_idx]

    return (
        y[sort_idx], x[sort_idx], *[sort_arg(arg) for arg in args],
    )
