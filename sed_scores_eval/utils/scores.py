import numpy as np
import pandas as pd


def extract_timestamps_and_classes_from_dataframe(scores, event_classes=None):
    column_names = list(scores.columns)
    assert len(column_names) > 2, column_names
    assert column_names[0] == 'onset', column_names
    assert column_names[1] == 'offset', column_names
    if event_classes is not None:
        assert column_names[2:] == event_classes, (column_names, event_classes)
    onset_times = scores['onset'].to_numpy()
    offset_times = scores['offset'].to_numpy()
    assert (offset_times[:-1] == onset_times[1:]).all(), (
        onset_times, offset_times)
    timestamps = np.concatenate((onset_times, offset_times[-1:]))
    return timestamps, column_names[2:]


def create_score_dataframe(scores, timestamps, event_classes):
    assert isinstance(scores, (np.ndarray, list, tuple)), type(scores)
    assert isinstance(timestamps, (np.ndarray, list, tuple)), type(timestamps)
    assert isinstance(event_classes, (list, tuple)), type(event_classes)
    scores = np.array(scores)
    timestamps = np.array(timestamps)
    assert timestamps.ndim == 1, timestamps.shape
    assert scores.shape == (len(timestamps)-1, len(event_classes)), (
        scores.shape, (len(timestamps)-1, len(event_classes)))
    assert all([
        isinstance(cls, (str, int)) for cls in event_classes
    ]), event_classes
    return pd.DataFrame(
        np.concatenate((
            timestamps[:-1, None], timestamps[1:, None], scores
        ), axis=1),
        columns=['onset', 'offset', *event_classes],
    )


def get_unique_thresholds(scores):
    """get thresholds lying between a unique score and next smaller unique score

    Args:
        scores (1d np.array): sed scores

    Returns:
        unique_thresholds (1d np.array): threshold values
        sort_indices (1d np.array): indices by which scores are sorted in
            ascending order
        unique_scores_indices (1d np.array): indices of the unique scores in
            the sorted scores array

    >>> score_arr = np.array([1,3,2,4,8,2])
    >>> get_unique_thresholds(score_arr)
    (array([-inf,  1.5,  2.5,  3.5,  6. ]), array([0, 2, 5, 1, 3, 4]), array([0, 1, 3, 4, 5]))
    """
    sort_indices = np.argsort(scores)
    sorted_scores = scores[sort_indices]
    unique_scores, unique_scores_indices = np.unique(
        sorted_scores, return_index=True
    )
    unique_thresholds = np.concatenate((
        [-np.inf], (unique_scores[1:] + unique_scores[:-1]) / 2
    ))
    return unique_thresholds, sort_indices, unique_scores_indices
