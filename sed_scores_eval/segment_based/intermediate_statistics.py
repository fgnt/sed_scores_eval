import numpy as np
from pathlib import Path
from sed_scores_eval.base_modules.io import parse_inputs, read_audio_durations
from sed_scores_eval.utils.scores import validate_score_dataframe
from sed_scores_eval.utils.array_ops import get_first_index_where
from sed_scores_eval.base_modules.ground_truth import multi_label_to_single_label_ground_truths


def intermediate_statistics(
        scores, ground_truth, audio_durations, *,
        segment_length=1., time_decimals=6,
):
    """

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        audio_durations: The duration of each audio file in the evaluation set.
        segment_length: the segment length of the segments that are to be
            evaluated.
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high detected or ground truth events that have
            onsets or offsets right on a segment boundary may swap over to the
            adjacent segment because of small deviations due to limited
            floating point precision.

    Returns:

    """
    scores, ground_truth, keys = parse_inputs(scores, ground_truth)
    if isinstance(audio_durations, (str, Path)):
        audio_durations = Path(audio_durations)
        assert audio_durations.is_file(), audio_durations
        audio_durations = read_audio_durations(audio_durations)
    single_label_ground_truths = None
    segment_scores = None
    segment_targets = None
    event_classes = None
    for audio_id in scores.keys():
        scores_k = scores[audio_id]
        timestamps, event_classes = validate_score_dataframe(
            scores_k, event_classes=event_classes)
        timestamps = np.round(timestamps, time_decimals)
        if single_label_ground_truths is None:
            single_label_ground_truths = multi_label_to_single_label_ground_truths(
                ground_truth, event_classes)
            segment_scores = {class_name: [] for class_name in event_classes}
            segment_targets = {class_name: [] for class_name in event_classes}
        scores_k = scores_k[event_classes].to_numpy()
        if audio_durations is None:
            duration = max(
                [timestamps[-1]] + [t_off for t_on, t_off, _ in ground_truth[audio_id]]
            )
        else:
            duration = audio_durations[audio_id]
        n_segments = int(np.ceil(duration / segment_length))
        segment_boundaries = np.round(
            np.arange(n_segments+1) * segment_length,
            time_decimals
        )
        segment_onsets = segment_boundaries[:-1]
        segment_offsets = segment_boundaries[1:]
        for class_name in event_classes:
            gt = single_label_ground_truths[class_name][audio_id]
            if len(gt) == 0:
                segment_targets[class_name].append(
                    np.zeros(n_segments, dtype=np.bool_))
            else:
                segment_targets[class_name].append(
                    np.any([
                        (segment_onsets < gt_offset)
                        * (segment_offsets > gt_onset)
                        * (segment_offsets > segment_onsets)
                        for gt_onset, gt_offset in
                        single_label_ground_truths[class_name][audio_id]
                    ], axis=0)
                )
        for i in range(n_segments):
            idx_on = get_first_index_where(
                timestamps, "gt", segment_onsets[i]) - 1
            idx_on = max(idx_on, 0)
            idx_off = get_first_index_where(
                timestamps, "geq", segment_offsets[i])
            idx_off = min(idx_off, len(timestamps)-1)
            if idx_off <= idx_on:
                scores_ki = np.zeros(scores_k.shape[-1])
            else:
                scores_ki = np.max(scores_k[idx_on:idx_off], axis=0)
            for c, class_name in enumerate(event_classes):
                segment_scores[class_name].append(scores_ki[c])
    stats = {}
    for class_name in event_classes:
        segment_scores[class_name] = np.array(segment_scores[class_name]+[np.inf])
        sort_idx = np.argsort(segment_scores[class_name])
        segment_scores[class_name] = segment_scores[class_name][sort_idx]
        segment_targets[class_name] = np.concatenate(
            segment_targets[class_name]+[np.zeros(1)])[sort_idx]
        tps = np.cumsum(segment_targets[class_name][::-1])[::-1]
        n_sys = np.arange(len(tps))[::-1]
        segment_scores[class_name], unique_idx = np.unique(segment_scores[class_name], return_index=True)
        stats[class_name] = {
            'tps': tps[unique_idx],
            'fps': n_sys[unique_idx] - tps[unique_idx],
            'n_ref': tps[0],
        }
    return {
        class_name: (segment_scores[class_name], stats[class_name])
        for class_name in event_classes
    }
