import numpy as np
import pandas as pd
from sed_scores_eval.utils.array_ops import get_first_index_where


def create_score_dataframe(scores, timestamps, event_classes):
    """compose SED scores with event class labels into pandas.DataFrame with
    corresponding frame onset and offset times

    Args:
        scores (2d np.array): (T, K) SED scores for multiple event classes,
            with T being the number of frames and K being the number of event
            classes.
        timestamps (1d np.array): onset timestamps for each frame plus one more
            timestamp which is the final offset time.
        event_classes (list of str): list of event class names

    Returns: pandas.DataFrame with one row per frame where first and second
        column are the frame onset and offset time and the other columns the
        detection scores for the various event classes

    """
    if not isinstance(scores, (np.ndarray, list, tuple)):
        raise ValueError(
            f'scores must be np.ndarray, list or tuple but {type(scores)} '
            f'was given.'
        )
    scores = np.asanyarray(scores)
    if scores.ndim != 2:
        raise ValueError(
            f'scores must be two-dimensional but has shape {scores.shape}'
        )
    if not isinstance(timestamps, (np.ndarray, list, tuple)):
        raise ValueError(
            f'timestamps must be np.ndarray, list or tuple but '
            f'{type(timestamps)} was given.'
        )
    timestamps = np.asanyarray(timestamps)
    if timestamps.shape != (len(scores)+1,):
        raise ValueError(
            f'timestamps must be one-dimensional and have length '
            f'len(scores)+1 ({len(scores) + 1}) but timestamps with shape '
            f'{timestamps.shape} was given.'
        )

    if not isinstance(event_classes, (list, tuple)):
        raise ValueError(
            f'event_classes must be list or tuple but {type(event_classes)} '
            f'was given.'
        )
    if len(event_classes) != scores.shape[1]:
        raise ValueError(
            f'length of event_classes ({len(event_classes)}) does not match '
            f'scores.shape[1]. scores.shape: {scores.shape}.'
        )
    if not all([isinstance(c, (str, int)) for c in event_classes]):
        raise ValueError(
            f'All event classes must be either str or int but '
            f'event_classes={event_classes} was given.'
        )
    return pd.DataFrame(
        np.concatenate((
            timestamps[:-1, None], timestamps[1:, None], scores
        ), axis=1),
        columns=['onset', 'offset', *event_classes],
    )


def validate_score_dataframe(scores, timestamps=None, event_classes=None):
    """validate that scores is a pandas.DataFrame and has the correct format
    namely as provided by create_score_dataframe and return timestamps array
    and list of event class names.

    Args:
        scores: SED scores
        timestamps:
        event_classes (list of str): optional list of event classes used to
            assert correct event labels in scores DataFrame

    Returns:

    """
    if not isinstance(scores, pd.DataFrame):
        raise ValueError(
            f'scores must be pandas.DataFrame but {type(scores)} was given.')
    column_names = list(scores.columns)
    if (
        len(column_names) < 3
        or column_names[0] != 'onset'
        or column_names[1] != 'offset'
    ):
        raise ValueError(
            f'scores must contain at least 3 columns with first and second '
            f'column being frame onset and offset time, respectively, and '
            f'subsequent columns being score columns for various event '
            f'classes. However, provided columns are {column_names}.'
        )
    if event_classes is not None and column_names[2:] != event_classes:
        raise ValueError(
            f'column names {column_names[2:]} do not match the event class '
            f'names {event_classes}'
        )
    onset_times = scores['onset'].to_numpy()
    offset_times = scores['offset'].to_numpy()
    timestamps_from_df = np.concatenate((onset_times, offset_times[-1:]))
    if (offset_times == onset_times).any():
        raise ValueError(f'Some frames have zero length: {timestamps_from_df}')
    if not (offset_times[:-1] == onset_times[1:]).all():
        raise ValueError(
            f'onset times must match offset times of the previous frame.'
        )
    if timestamps is not None and not np.allclose(timestamps_from_df, timestamps):
        raise ValueError(
            f'timestamps from file {timestamps_from_df} do not match provided timestamps {timestamps}.'
        )
    event_classes = column_names[2:]
    for event_class in event_classes:
        assert "\\" not in event_class, ('class names must not contain "\\"', event_class)
    return timestamps_from_df, event_classes


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


def onset_offset_times_to_score_indices(onset_time, offset_time, timestamps):
    """

    Args:
        onset_time:
        offset_time:
        timestamps:

    Returns:
        onset_idx:
        offset_idx:

    """
    assert offset_time > onset_time, (onset_time, offset_time)
    # assert t_off <= timestamps[-1], (t_off, timestamps[-1])
    onset_idx = max(
        get_first_index_where(timestamps, 'gt', onset_time) - 1,
        0
    )
    assert timestamps[onset_idx] <= onset_time, (
        timestamps[onset_idx], onset_time)
    offset_idx = min(
        get_first_index_where(timestamps, 'geq', offset_time),
        len(timestamps) - 1,
    )
    assert offset_idx > onset_idx, (onset_idx, offset_idx)
    # assert timestamps[onset_idx] <= onset_time < timestamps[onset_idx+1]
    # assert timestamps[offset_idx-1] < offset_time <= timestamps[onset_idx+1]
    return onset_idx, offset_idx
