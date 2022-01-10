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
        x = np.concatenate((x[:cutoff_idx], [max_x]))
        y = y[:cutoff_idx]
    widths = x[1:] - x[:-1]
    areas = y * widths
    auc = sum(areas)
    return auc
