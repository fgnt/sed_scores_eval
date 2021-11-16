import numpy as np
from sed_scores_eval.utils.array_ops import cummin
from einops import rearrange


def detection_onset_offset_times(scores, timestamps):
    """

    Args:
        scores:
        timestamps:

    Returns:

    >>> y = np.array([.4,1.,.1,.6,.5,.6,.4,.3,.6,.2])
    >>> ts = np.linspace(0.,len(y)*.2,len(y) + 1)
    >>> detection_onset_offset_times(y, ts)
    """
    onset_deltas_ = onset_deltas(scores)
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
        onset_offset_times.append(_single_detection_onset_offset_times(
            scores, timestamps, current_spawn_idx, current_merge_idx))
    onset_offset_times = np.array(onset_offset_times)[:, :, unique_indices]
    onset_offset_times = rearrange(onset_offset_times, 'd t b -> t b d')
    onset_times, offset_times = onset_offset_times
    return scores_unique, onset_times, offset_times


def _single_detection_onset_offset_times(
        scores, timestamps, spawn_idx, merge_idx):
    """

    Args:
        scores:
        timestamps:
        spawn_idx:
        merge_idx:

    Returns:

    >>> y = np.array([.4,1.,.1,.6,.5,.6,.4,.3,.6,.2])
    >>> ts = np.linspace(0.,len(y)*.2,len(y) + 1)
    >>> _single_detection_onset_offset_times(y, ts, 1, 0)
    >>> _single_detection_onset_offset_times(y, ts, 3, 0)
    >>> _single_detection_onset_offset_times(y, ts, 3, 2)
    >>> _single_detection_onset_offset_times(y, ts, 5, 4)
    >>> _single_detection_onset_offset_times(y, ts, 8, 0)
    >>> _single_detection_onset_offset_times(y, ts, 8, 7)
    """
    pre_spawn_cummin_indices = np.unique(spawn_idx - cummin(scores[:spawn_idx+1][::-1])[1])
    post_spawn_cummin_indices = np.unique(spawn_idx + cummin(scores[spawn_idx:])[1])
    onset_time_indices = [0] + (pre_spawn_cummin_indices[:-1]+1).tolist()
    onset_times = timestamps[onset_time_indices]
    offset_time_indices = (post_spawn_cummin_indices[1:]).tolist()+[len(scores)]
    offset_times = timestamps[offset_time_indices]

    onset_time_deltas = np.zeros_like(scores)
    onset_time_deltas[pre_spawn_cummin_indices[:-1]] = onset_times[:-1] - onset_times[1:]
    offset_time_deltas = np.zeros_like(scores)
    offset_time_deltas[post_spawn_cummin_indices] = offset_times - np.concatenate(
        ([timestamps[spawn_idx]], offset_times[:-1]))

    unique_scores, inverse_indices = np.unique(scores, return_inverse=True)
    unique_score_onset_time_deltas = np.zeros_like(unique_scores)
    np.add.at(unique_score_onset_time_deltas, inverse_indices, onset_time_deltas)
    unique_score_onset_times = timestamps[spawn_idx] + np.cumsum(unique_score_onset_time_deltas[::-1])[::-1]
    onset_times = unique_score_onset_times[inverse_indices]
    unique_score_offset_time_deltas = np.zeros_like(unique_scores)
    np.add.at(unique_score_offset_time_deltas, inverse_indices, offset_time_deltas)
    unique_score_offset_times = timestamps[spawn_idx] + np.cumsum(unique_score_offset_time_deltas[::-1])[::-1]
    offset_times = unique_score_offset_times[inverse_indices]

    if merge_idx > 0:
        onset_times[scores <= scores[merge_idx]] = timestamps[spawn_idx]
        offset_times[scores <= scores[merge_idx]] = timestamps[spawn_idx]
    return onset_times, offset_times


def onset_deltas(scores):
    """return the change in the total number of onsets when decision threshold
    falls below a score.

    Args:
        scores (1d nparray): soft sound event detection scores

    Returns:

    """
    assert isinstance(scores, np.ndarray), scores
    prev_scores = np.concatenate(([-np.inf], scores[:-1]))
    next_scores = np.concatenate((scores[1:], [-np.inf]))
    return (
            (scores > prev_scores).astype(np.int)
            - (next_scores > scores).astype(np.int)
    )
