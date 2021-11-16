import numpy as np
from sed_scores_eval.utils.scores import extract_timestamps_and_classes_from_dataframe
from sed_scores_eval.base_modules.ground_truth import multi_label_to_single_label_ground_truths
from sed_scores_eval.base_modules.detection import detection_onset_offset_times
from sed_scores_eval.base_modules.io import parse_inputs


def accumulated_intermediate_statistics(
        scores, ground_truth, intermediate_statistics_fn,
        **intermediate_statistics_fn_kwargs
):
    scores, ground_truth, audio_ids = parse_inputs(scores, ground_truth)

    _, event_classes = extract_timestamps_and_classes_from_dataframe(
        scores[audio_ids[0]])
    single_label_ground_truths = multi_label_to_single_label_ground_truths(
        ground_truth, event_classes)

    num_stats = None
    change_point_scores = None
    deltas = None
    for audio_id in audio_ids:
        scores_for_key = scores[audio_id]
        timestamps, _ = extract_timestamps_and_classes_from_dataframe(
            scores_for_key, event_classes)
        scores_for_key = scores_for_key[event_classes].to_numpy()
        gt_onset_times = []
        gt_offset_times = []
        for c, cls in enumerate(event_classes):
            gt = single_label_ground_truths[cls][audio_id]
            if gt:
                current_onset_times, current_offset_times = np.array(gt).T
            else:
                current_onset_times = current_offset_times = np.empty(0)
            gt_onset_times.append(current_onset_times)
            gt_offset_times.append(current_offset_times)
        for c, cls in enumerate(event_classes):
            (
                unique_scores, detection_onset_times, detection_offset_times
            ) = detection_onset_offset_times(scores_for_key[:, c], timestamps)
            stats = intermediate_statistics_fn(
                detection_onset_times=detection_onset_times,
                detection_offset_times=detection_offset_times,
                target_onset_times=gt_onset_times[c],
                target_offset_times=gt_offset_times[c],
                other_onset_times=gt_onset_times[:c] + gt_onset_times[c+1:],
                other_offset_times=gt_offset_times[:c] + gt_offset_times[c+1:],
                **intermediate_statistics_fn_kwargs,
            )
            if num_stats is None:
                num_stats = len(stats)
                change_point_scores = {cls: [] for cls in event_classes}
                deltas = {
                    cls: {key: [] for key in stats}
                    for cls in event_classes
                }
            cp_scores_c, deltas_c = deltas_from_intermediate_statistics(
                unique_scores, stats
            )
            change_point_scores[cls].append(cp_scores_c)
            for key in deltas_c:
                deltas[cls][key].append(deltas_c[key])

    return {
        cls: intermediate_statistics_from_deltas(
            np.concatenate(change_point_scores[cls]),
            {key: np.concatenate(deltas[cls][key]) for key in deltas[cls]}
        )
        for cls in event_classes
    }


def deltas_from_intermediate_statistics(scores, intermediate_stats):
    scores_unique, unique_idx = np.unique(scores, return_index=True)
    intermediate_stats = {
        key: stat[unique_idx] for key, stat in intermediate_stats.items()
    }
    deltas = {
        key: stat - np.concatenate((stat[1:], np.zeros_like(stat[:1])))
        for key, stat in intermediate_stats.items()
    }

    # filter scores where nothing changes
    any_delta = np.array([
        np.abs(d).sum(tuple([i for i in range(d.ndim) if i > 0]))
        for d in deltas.values()
    ]).sum(0) > 0
    change_indices = np.argwhere(any_delta).flatten()
    change_point_scores = scores_unique[change_indices]
    deltas = {key: deltas_i[change_indices] for key, deltas_i in deltas.items()}
    return change_point_scores, deltas


def intermediate_statistics_from_deltas(scores, deltas):
    scores_unique, inverse_idx = np.unique(scores, return_inverse=True)
    b = len(scores_unique)
    scores_unique = np.concatenate((scores_unique, [np.inf]))
    stats = {}
    for key, d in deltas.items():
        deltas_unique = np.zeros((b, *d.shape[1:]))
        np.add.at(deltas_unique, inverse_idx, d)
        stats[key] = np.concatenate((
            np.cumsum(deltas_unique[::-1], axis=0)[::-1], np.zeros_like(deltas_unique[:1])
        ))
    return scores_unique, stats
