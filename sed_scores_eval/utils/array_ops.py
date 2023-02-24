import numpy as np


def cummin(array):
    """cummulative minimum operation on 1d arrays

    Args:
        array:

    Returns:
        cummin_values (1d np.ndarray): cummin values
        cummin_indices (1d np.ndarray): indices of cummin values in input array

    >>> cummin(np.array([1.,3.,2.,0.,-1.,3,-2]))
    (array([ 1.,  1.,  1.,  0., -1., -1., -2.]), array([0, 0, 0, 3, 4, 4, 6]))
    >>> cummin(np.array([0.,0.,0.]))
    (array([0., 0., 0.]), array([0, 0, 0]))
    """
    cummin_values = np.minimum.accumulate(array)
    cummin_indices = np.concatenate((
        [0],
        np.arange(1, len(array))[cummin_values[1:] < cummin_values[:-1]]
    ))
    # _, cummin_indices = np.unique(cummin_values, return_index=True)
    return cummin_values, cummin_indices


def cummax(array):
    """cummulative maximum operation on 1d arrays

    Args:
        array:

    Returns:
        cummax_values (1d np.ndarray): cummax values
        cummax_indices (1d np.ndarray): indices of cummax values in input array

    >>> cummax(np.array([1.,3.,2.,0.,-1.,4.]))
    (array([1., 3., 3., 3., 3., 4.]), array([0, 1, 1, 1, 1, 5]))
    >>> cummax(np.array([0.,0.,0.]))
    (array([0., 0., 0.]), array([0, 0, 0]))
    """
    cummax_values = np.maximum.accumulate(array)
    cummax_indices = np.concatenate((
        [0],
        np.arange(1, len(array))[cummax_values[1:] > cummax_values[:-1]]
    ))
    # _, cummax_indices = np.unique(cummax_values, return_index=True)
    return cummax_values, cummax_indices


def get_first_index_where(array, criterion, value, *, axis=0):
    """get the first index where array fulfills a criterion w.r.t. value,
    where criterion may be "geq" (greater equal), "gt" (greater than),
    "leq" (less equal) or "lt" (less than). If criterion is met nowhere,
    function returns len(array)

    Args:
        array (1d np.ndarray):
        criterion (str):
        value (number):

    Returns:
        idx: first index where criterion is met or len(array) if criterion is
            never met.

    >>> arr = np.array([1,2,3,4,5])
    >>> get_first_index_where(arr, "geq", 3)
    2
    >>> get_first_index_where(arr, "gt", 3)
    3
    >>> get_first_index_where(arr, "gt", 5)
    5
    """
    if criterion == "geq":
        bool_idx = array >= value
    elif criterion == "gt":
        bool_idx = array > value
    elif criterion == "leq":
        bool_idx = array <= value
    elif criterion == "lt":
        bool_idx = array < value
    else:
        raise ValueError(f'Invalid criterion {criterion}')
    concat_shape = list(bool_idx.shape)
    concat_shape[axis] = 1
    bool_idx = np.concatenate((bool_idx, np.ones(concat_shape, dtype=bool)), axis=axis)
    return np.argmax(bool_idx, axis=axis)
