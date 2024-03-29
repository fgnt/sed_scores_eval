# distutils: language = c++
#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from numpy.math cimport INFINITY


def onset_offset_curves(scores_in, timestamps_in, change_point_candidates_in=None):
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

    Args:
        scores_in (1d np.ndarray): SED scores for a single event class
        timestamps_in (1d np.ndarray): onset timestamps for each score plus one more
            timestamp which is the final offset time.
        change_point_candidates_in (1d np.ndarray):

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
    scores_in = np.asanyarray(scores_in, dtype=np.float64)
    if not scores_in.ndim == 1:
        raise ValueError(
            f'scores must be 1-dimensional array of single class SED scores, '
            f'but array of shape {scores_in.shape} was given.'
        )
    timestamps_in = np.asanyarray(timestamps_in, dtype=np.float64)
    if not timestamps_in.ndim == 1 or len(timestamps_in) != (len(scores_in) + 1):
        raise ValueError(
            f'timestamps must be 1-dimensional array of length(len(scores) + 1), '
            f'but array of shape {timestamps_in.shape} was given.'
        )

    if change_point_candidates_in is None:
        change_point_candidates_in = np.unique(scores_in)
    change_point_candidates_in = np.asanyarray(change_point_candidates_in, dtype=np.float64)

    cdef double [:] scores = scores_in
    cdef double [:] timestamps = timestamps_in
    cdef double [:] change_point_candidates = change_point_candidates_in

    cdef int num_segments = len(scores)
    cdef int num_candidates = len(change_point_candidates)
    cdef int max_events = 0

    cdef double [:,:] onset_times = np.zeros((len(change_point_candidates), (num_segments+1)//2))
    cdef double [:,:] offset_times = np.zeros_like(onset_times)

    cdef:
        int i, t, event_idx
        double threshold

    for i in range(num_candidates):
        threshold = change_point_candidates[i]
        event_idx = 0
        is_on = False
        for t in range(num_segments):
            if is_on:
                if scores[t] < threshold or scores[t] == -INFINITY:
                    offset_times[i, event_idx] = timestamps[t]
                    event_idx += 1
                    is_on = False
            else:
                if scores[t] >= threshold and scores[t] > -INFINITY:
                    onset_times[i, event_idx] = timestamps[t]
                    is_on = True
        if is_on:
            offset_times[i, event_idx] = timestamps[num_segments]
            event_idx += 1
        max_events = max(max_events, event_idx)
    return np.array(change_point_candidates), np.array(onset_times[:, :max_events]), np.array(offset_times[:, :max_events])
