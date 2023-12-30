import numpy as np
from numbers import Number
from sed_scores_eval.utils.array_ops import get_first_index_where


def multi_label_to_single_label_ground_truths(ground_truth, event_classes):
    """convert dict of multi label ground truths to list of dict of single
    label ground truths.

    Args:
        ground_truth (dict of lists of tuples): list of ground truth event
            tuples (onset, offset, event label) for each audio clip.
        event_classes (list of str): list of event classes.

    Returns (dict of dicts of lists of tuple): list of single class ground
        truth event tuples (onset, offset) for each audio clip for each event
        class

    """
    if not isinstance(ground_truth, dict):
        raise ValueError(
            f'ground_truth must be of type dict but {type(ground_truth)} was '
            f'given.'
        )
    single_label_ground_truths = {
        class_name: {} for class_name in event_classes}
    n_ref_count = {
        class_name: 0 for class_name in event_classes}
    for clip_id in ground_truth.keys():
        for class_name in event_classes:
            single_label_ground_truths[class_name][clip_id] = []
        if not isinstance(ground_truth[clip_id], (list, tuple)):
            raise ValueError(
                f'ground_truth values must be of type list/tuple but '
                f'{type(ground_truth[clip_id])} was found for key {clip_id}.'
            )
        for i in range(len(ground_truth[clip_id])):
            if (
                not isinstance(ground_truth[clip_id][i], (list, tuple))
                or len(ground_truth[clip_id][i]) != 3
            ):
                raise ValueError(
                    f'ground_truth event tuples must have the form '
                    f'(onset_time, offset_time, event label) but '
                    f'{ground_truth[clip_id][i]} was given for key {clip_id}.'
                )
            t_on, t_off, event_label = ground_truth[clip_id][i]
            if (
                not isinstance(t_on, Number) or not isinstance(t_off, Number)
                or not isinstance(event_label, (str, int))
            ):
                raise ValueError(
                    f'ground_truth event tuples must have the form '
                    f'(onset time, offset time, event label) with onset time '
                    f'and offset time being numbers and event label either '
                    f'being integer or string but types {type(t_on)}, '
                    f'{type(t_off)} and {type(event_label)} were given for '
                    f'key {clip_id}.'
                )
            if event_label not in event_classes:
                raise ValueError(
                    f'event label {event_label} for key {clip_id} is not listed '
                    f'in event_classes {event_classes}.'
                )
            single_label_ground_truths[event_label][clip_id].append((t_on, t_off))
            n_ref_count[event_label] += 1
    classes_without_ground_truth_events = [
        class_name for class_name, n_ref in n_ref_count.items() if n_ref == 0
    ]
    assert len(classes_without_ground_truth_events) == 0, f"No ground truth events for classes {classes_without_ground_truth_events}."
    return single_label_ground_truths


def event_counts_and_durations(ground_truth, event_classes):
    """obtain the total number and combined duration of ground truths events
    for each event class.

    Args:
        ground_truth (dict of lists of tuples): list of ground truth event
            tuples (onset, offset, event label) for each audio clip.
        event_classes (list of str): list of event classes.

    Returns:
        counts (dict of int): total number of ground truth events for each
            event class
        durations (dict of float): combined duration of ground truth events for
            each event class

    """
    if not isinstance(ground_truth, dict):
        raise ValueError(
            f'ground_truth must be of type dict but {type(ground_truth)} was'
            f'given.'
        )
    counts = {key: 0 for key in event_classes}
    durations = {key: 0. for key in event_classes}
    for key in ground_truth.keys():
        if not isinstance(ground_truth[key], (list, tuple)):
            raise ValueError(
                f'ground_truth values must be of type list/tuple but '
                f'{type(ground_truth[key])} was found for key {key}.'
            )
        for i in range(len(ground_truth[key])):
            if (
                not isinstance(ground_truth[key][i], (list, tuple))
                or len(ground_truth[key][i]) != 3
            ):
                raise ValueError(
                    f'ground_truth event tuples must have the form '
                    f'(onset_time, offset_time, event label) but '
                    f'{ground_truth[key][i]} was given for key {key}.'
                )
            t_on, t_off, event_label = ground_truth[key][i]
            if (
                not isinstance(t_on, Number) or not isinstance(t_off, Number)
                or not isinstance(event_label, (str, int))
            ):
                raise ValueError(
                    f'ground_truth event tuples must have the form '
                    f'(onset time, offset time, event label) with onset time '
                    f'and offset time being numbers and event label either '
                    f'being integer or string but types {type(t_on)}, '
                    f'{type(t_off)} and {type(event_label)} were given for '
                    f'key {key}.'
                )
            if event_label not in event_classes:
                raise ValueError(
                    f'event label {event_label} for key {key} is not listed '
                    f'in event_classes {event_classes}.'
                )
            counts[event_label] += 1
            durations[event_label] += t_off - t_on
    return counts, durations


def onset_offset_times_to_indices(onset_time, offset_time, timestamps):
    """convert an onset/offset time pair to the indices of the frames in which
    the onset/offset time lie (or the last frame index if the offset_time lies
    beyond timestamps[-1]).

    Args:
        onset_time (float):
        offset_time (float):
        timestamps (1d np.array): onset timestamps for each frame plus one more
            timestamp which is the final offset time.

    Returns:
        onset_idx:
        offset_idx:

    """
    if not np.isscalar(onset_time) or not np.isscalar(offset_time):
        raise ValueError(
            f'onset_time and offset_time must be scalars, but '
            f'{type(onset_time)} and {type(offset_time)} were given.'
        )
    timestamps = np.asanyarray(timestamps)
    if not timestamps.ndim == 1:
        raise ValueError(
            f'timestamps must be 1-dimensional array but array of shape '
            f'{timestamps.shape} was given.'
        )
    if not offset_time > onset_time >= timestamps[0] >= 0.:
        raise ValueError(
            f'offset_time must be greater than onset_time which must be '
            f'greater equal timestamps[0] which must be greater equal 0. '
            f'However, offset time is {offset_time}, onset time is '
            f'{onset_time} and timestamps are {timestamps}.'
        )
    onset_idx = max(
        get_first_index_where(timestamps, 'gt', onset_time) - 1,
        0
    )
    offset_idx = min(
        get_first_index_where(timestamps, 'geq', offset_time),
        len(timestamps) - 1,
    )
    return onset_idx, offset_idx
