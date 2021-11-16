from sed_scores_eval.utils.array_ops import get_first_index_where


def multi_label_to_single_label_ground_truths(ground_truth, event_classes):
    single_label_ground_truths = {cls: {} for cls in event_classes}
    for key in ground_truth.keys():
        for cls in event_classes:
            single_label_ground_truths[cls][key] = []
        for t_on, t_off, event_class in ground_truth[key]:
            assert event_class in event_classes, (event_classes, event_class)
            single_label_ground_truths[event_class][key].append((t_on, t_off))
    return single_label_ground_truths


def event_counts_and_durations(ground_truth, event_classes):
    counts = {key: 0 for key in event_classes}
    durations = {key: 0. for key in event_classes}
    for gt in ground_truth.values():
        for t_on, t_off, event_label in gt:
            assert event_label in counts, (event_label, counts.keys())
            assert event_label in durations, (event_label, durations.keys())
            counts[event_label] += 1
            durations[event_label] += t_off - t_on
    return counts, durations


def onset_offset_times_to_indices(onset_time, offset_time, timestamps):
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
