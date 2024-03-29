import numpy as np
import pandas as pd
from sed_scores_eval.base_modules.scores import validate_score_dataframe


def onset_offset_curves(scores, timestamps, change_point_candidates=None):
    """get onset and offset times of event detections for various decision
    thresholds. Here, the number of event detections is given by the number of
    local maxima in the score signal with events being spawned when the
    decision threshold falls below the local maximum. However, usually only a
    subset of these events is active simultanously while others are inactive,
    e.g., because a certain threshold does not yet fall below all local
    maximums. For inactive events we return offset_time = onset_time. Further,
    when the decision threshold falls below a local minimum, two separate
    events merge into a single event. In this case, we keep the earlier event
    active with corresponding onset and offset times, while the later event is
    set inactive with offset_time = onset_time.

    ! There is a faster implementation of this function in cy_detection.pyx !

    Args:
        scores (1d np.ndarray): SED scores for a single event class
        timestamps (1d np.ndarray): onset timestamps for each score plus one more
            timestamp which is the final offset time.
        change_point_candidates (1d np.ndarray)

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
            [0. , 0. ]],
    <BLANKLINE>
           [[0.2, 1.2],
            [0. , 0. ]],
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

    if change_point_candidates is None:
        change_point_candidates = np.unique(scores)
    detection = scores >= change_point_candidates[:, None]
    prev_detection = np.concatenate(
        (np.zeros_like(detection[:, :1]), detection),
        axis=1,
    )
    detection = np.concatenate(
        (detection, np.zeros_like(detection[:, :1]),),
        axis=1,
    )
    onsets = detection > prev_detection
    offsets = detection < prev_detection
    # assert (onsets.sum(-1) == offsets.sum(-1)).all()
    n_events = onsets.sum(-1)
    max_events = n_events.max()

    onset_times = np.zeros((len(change_point_candidates), max_events))
    offset_times = np.zeros_like(onset_times)
    thres_indices, frame_indices = np.argwhere(onsets).T
    n_events_offset = np.cumsum(n_events) - n_events
    event_indices = np.arange(len(thres_indices)) - n_events_offset[thres_indices]
    onset_times[thres_indices, event_indices] = timestamps[frame_indices]
    thres_indices, frame_indices = np.argwhere(offsets).T
    offset_times[thres_indices, event_indices] = timestamps[frame_indices]
    # print(onset_times.shape)
    return change_point_candidates, onset_times, offset_times


def onset_deltas(scores):
    """return the change in the total number of onsets when decision threshold
    falls below each of the scores, i.e., +1 at local maximums and -1 at local
    minimums in score signal.

    Args:
        scores (1d np.ndarray): SED scores for a single event class

    Returns:
        onset_deltas (1d np.ndarray): array with same length as scores
        indicating the change in the number of onsets when decision threshold
        falls below each of the scores, i.e., +1 at local maxima and -1 at
        local minima in score signal.

    >>> onset_deltas(np.array([1,2,3,3,4,3]))
    """
    assert isinstance(scores, np.ndarray), scores
    prev_scores = np.concatenate(([-np.inf], scores[:-1]))
    next_scores = np.concatenate((scores[1:], [-np.inf]))
    return (
        (scores > prev_scores).astype(int)
        - (next_scores > scores).astype(int)
    ) * (scores > -np.inf)


def scores_to_event_list(scores, thresholds, event_classes=None):
    """detect events and return as list

    Args:
        scores ((dict of) pandas.DataFrame): containing onset and offset times
            of a score window in first two columns followed by sed score
            columns for each event class.
        thresholds (1d np.ndarray or dict of floats): thresholds to be used for
            the different event classes.
        event_classes (list of str): optional list of event classes used to
            assert correct event labels in scores DataFrame

    Returns:
        event_list (list of tuple): list of events as tuples (onset, offset, event_class)
    """
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
    detections = np.concatenate((zeros, detections, zeros), axis=0).astype(float)
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
