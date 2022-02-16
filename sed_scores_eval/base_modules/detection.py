import numpy as np
import pandas as pd
from einops import rearrange
from sed_scores_eval.utils.array_ops import cummin
from sed_scores_eval.utils.scores import validate_score_dataframe


def onset_offset_curves(scores, timestamps):
    """get onset and offset times of event detections for various decision
    thresholds. Here, the number of event detections is given by the number of
    local maximums in the score signal with events being spawned when the
    decision threshold falls below the local maximum. However, usually only a
    subset of these events is active simultanously while others are inactive,
    e.g., because a certain threshold does not yet fall below all local
    maximums. For inactive events we return offset_time = onset_time. Further,
    when the decision threshold falls below a local minimum, two separate
    events merge into a single event. In this case, we keep the earlier event
    active with corresponding onset and offset times, while the later event is
    set inactive with offset_time = onset_time.

    Args:
        scores (1d np.ndarray): SED scores for a single event class
        timestamps (1d np.ndarray): onset timestamps for each score plus one more
            timestamp which is the final offset time.

    Returns:
        scores_unique (1d np.ndarray): unique and sorted score array
        onset_times (2d np.ndarray): onset times for each possible event at
            each decsion threshold that falls below one of the unique scores.
            Shape is (number of unique scores, number of events/local maximums).
        offset_times (2d np.ndarray): offset times for each possible event at
            each decsion threshold that falls below one of the unique scores.
            Shape is (number of unique scores, number of events/local maximums).

    >>> y = np.array([.4,1.,.6,.6,1.,1.,.4])
    >>> ts = np.linspace(0., len(y)*.2, len(y) + 1)  # each score has width of 200ms
    >>> y, t_on, t_off = onset_offset_curves(y, ts)
    >>> y
    array([0.4, 0.6, 1. ])
    >>> np.stack((t_on, t_off), axis=-1)
    array([[[0. , 1.4],
            [0.8, 0.8]],
    <BLANKLINE>
           [[0.2, 1.2],
            [0.8, 0.8]],
    <BLANKLINE>
           [[0.2, 0.4],
            [0.8, 1.2]]])
    """
    scores = np.asanyarray(scores)
    if not scores.ndim == 1:
        raise ValueError(
            f'scores must be 1-dimensional array of single class SED scores, '
            f'but array of shape {scores.shape} was given.'
        )
    timestamps = np.asanyarray(timestamps)
    if not timestamps.ndim == 1 or len(timestamps) != (len(scores) + 1):
        raise ValueError(
            f'timestamps must be 1-dimensional array of length(len(scores) + 1), '
            f'but array of shape {timestamps.shape} was given.'
        )

    onset_deltas_ = _onset_deltas(scores)
    event_spawn_indices = np.argwhere(onset_deltas_ > 0.5).flatten()
    event_merge_indices = np.argwhere(onset_deltas_ < -0.5).flatten()
    assert len(event_merge_indices) == len(event_spawn_indices) - 1, (
        len(event_spawn_indices), len(event_merge_indices))
    assert (event_spawn_indices[:-1] < event_merge_indices).all(), (
        event_spawn_indices, event_merge_indices)
    scores_unique, unique_indices, inverse_indices = np.unique(
        scores, return_index=True, return_inverse=True)
    onset_offset_times = []
    for i, current_spawn_idx in enumerate(event_spawn_indices):
        if i == 0:
            current_merge_idx = 0
        else:
            current_merge_idx = event_merge_indices[i-1]
        onset_offset_times.append(_single_detection_onset_offset_curve(
            scores, timestamps, current_spawn_idx, current_merge_idx,
            scores_unique, inverse_indices
        ))
    onset_offset_times = np.array(onset_offset_times)
    onset_offset_times = rearrange(onset_offset_times, 'd t b -> t b d')
    onset_times, offset_times = onset_offset_times
    return scores_unique, onset_times, offset_times


def _single_detection_onset_offset_curve(
        scores, timestamps, spawn_idx, merge_idx_prev,
        scores_unique, inverse_indices,
):
    """get onset and offset times when threshold falls below each of the scores
    for a single event that is spawned when threshold falls below the local
    maximum at spawn_idx and is merged with the previous event when threshold
    falls below the local minimum at merge_idx. merge_idx == 0 indicates that
    event is the first event / local maximum.

    Args:
        scores (1d np.ndarray): SED scores for a single event class
        timestamps (1d np.ndarray): onset timestamps for each score plus one more
            timestamp which is the final offset time.
        spawn_idx (int): Index of local maximum
        merge_idx_prev (int): Index of previous local minimum. If merge_idx == 0
            event is considered the first event / local maximum.
        scores_unique (1d np.ndarray):
        inverse_indices (1d np.ndarray):

    Returns:
        onset_times (1d np.ndarray): onset times for current event when decsion
            threshold falls below each of the scores.
        offset_times (1d np.ndarray): offset times for current event when
            decsion threshold falls below each of the scores.

    >>> y = np.array([.4,1.,.1,.6,.5,.6,.4,])
    >>> ts = np.linspace(0.,len(y)*.2,len(y) + 1)  # each score has width of 200ms
    >>> _single_detection_onset_offset_curve(y, ts, 1, 0)
    (array([0. , 0.2, 0. , 0.2, 0.2, 0.2, 0. ]), array([0.4, 0.4, 1.4, 0.4, 0.4, 0.4, 0.4]))
    >>> _single_detection_onset_offset_curve(y, ts, 3, 2)
    (array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]), array([1.4, 0.6, 0.6, 0.8, 1.2, 0.8, 1.4]))
    >>> _single_detection_onset_offset_curve(y, ts, 5, 4)
    (array([1., 1., 1., 1., 1., 1., 1.]), array([1. , 1. , 1. , 1.2, 1. , 1.2, 1. ]))
    """
    # pre_spawn_cummin_indices = np.unique(spawn_idx - cummin(scores[merge_idx:spawn_idx+1][::-1])[1])
    pre_spawn_cummin_indices = np.arange(merge_idx_prev, spawn_idx + 1)
    post_spawn_cummin_indices = spawn_idx + cummin(scores[spawn_idx:])[1]

    onset_time_indices = [0] + (pre_spawn_cummin_indices[:-1]+1).tolist()
    onset_times = timestamps[onset_time_indices]
    offset_time_indices = (post_spawn_cummin_indices[1:]).tolist()+[len(scores)]
    offset_times = timestamps[offset_time_indices]

    onset_time_deltas = np.zeros_like(scores_unique)
    onset_time_deltas[inverse_indices[pre_spawn_cummin_indices[:-1]]] = onset_times[:-1] - onset_times[1:]
    offset_time_deltas = np.zeros_like(scores_unique)
    offset_time_deltas[inverse_indices[post_spawn_cummin_indices]] = offset_times - np.concatenate(
        ([timestamps[spawn_idx]], offset_times[:-1]))

    onset_times = timestamps[spawn_idx] + np.cumsum(onset_time_deltas[::-1])[::-1]
    offset_times = timestamps[spawn_idx] + np.cumsum(offset_time_deltas[::-1])[::-1]

    if merge_idx_prev > 0:
        onset_times[scores_unique <= scores[merge_idx_prev]] = timestamps[spawn_idx]
        offset_times[scores_unique <= scores[merge_idx_prev]] = timestamps[spawn_idx]
    return onset_times, offset_times


def _onset_deltas(scores):
    """return the change in the total number of onsets when decision threshold
    falls below each of the scores, i.e., +1 at local maximums and -1 at local
    minimums in score signal.

    Args:
        scores (1d np.ndarray): SED scores for a single event class

    Returns:
        onset_deltas (1d np.ndarray): array with same length as scores
        indicating the change in the number of onsets when decision threshold
        falls below each of the scores, i.e., +1 at local maximums and -1 at
        local minimums in score signal.

    >>> _onset_deltas(np.array([1,2,3,3,4,3]))
    """
    assert isinstance(scores, np.ndarray), scores
    prev_scores = np.concatenate(([-np.inf], scores[:-1]))
    next_scores = np.concatenate((scores[1:], [-np.inf]))
    return (
        (scores > prev_scores).astype(np.int)
        - (next_scores > scores).astype(np.int)
    )


def scores_to_event_list(scores, thresholds, event_classes=None):
    if not isinstance(scores, pd.DataFrame) and hasattr(scores, 'keys'):
        assert callable(scores.keys)
        keys = sorted(scores.keys())
        _, event_classes = validate_score_dataframe(
            scores[keys[0]], event_classes=event_classes)
        if isinstance(thresholds, dict):
            thresholds = np.array([
                thresholds[event_class] for event_class in event_classes])
        return {
            key: scores_to_event_list(
                scores[key], thresholds, event_classes=event_classes)
            for key in keys
        }
    timestamps, event_classes = validate_score_dataframe(
        scores, event_classes=event_classes)
    onset_times = scores['onset'].to_numpy()
    offset_times = scores['offset'].to_numpy()
    scores = scores[event_classes].to_numpy()
    if isinstance(thresholds, dict):
        thresholds = np.array([
            thresholds[event_class] for event_class in event_classes])
    detections = scores > thresholds
    zeros = np.zeros_like(detections[:1, :])
    detections = np.concatenate((zeros, detections, zeros), axis=0).astype(np.float)
    change_points = detections[1:] - detections[:-1]
    event_list = []
    for k in np.argwhere(np.abs(change_points).max(0) > .5).flatten():
        onsets = np.argwhere(change_points[:, k] > .5).flatten()
        offsets = np.argwhere(change_points[:, k] < -.5).flatten()
        assert len(onsets) == len(offsets) > 0
        for onset, offset in zip(onsets, offsets):
            event_list.append((
                onset_times[onset], offset_times[offset-1],
                event_classes[k]
            ))
    return sorted(event_list)
