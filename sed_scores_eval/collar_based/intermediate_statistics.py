import numpy as np
from sed_eval.util import bipartite_match
from sed_scores_eval.base_modules.ground_truth import event_counts_and_durations
from sed_scores_eval.base_modules.statistics import accumulated_intermediate_statistics
from sed_scores_eval.base_modules.io import parse_inputs


def intermediate_statistics(
    scores, ground_truth, onset_collar, offset_collar, offset_collar_rate=0.,
    time_decimals=6,
):
    scores, ground_truth, _ = parse_inputs(scores, ground_truth)

    multi_label_statistics = accumulated_intermediate_statistics(
        scores, ground_truth,
        intermediate_statistics_fn=statistics_fn,
        onset_collar=onset_collar, offset_collar=offset_collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals,
    )
    n_ref, _ = event_counts_and_durations(
        ground_truth, event_classes=multi_label_statistics.keys()
    )
    return {
        cls: (cp_scores_cls, {**stats_cls, 'n_ref': n_ref[cls]})
        for cls, (cp_scores_cls, stats_cls) in multi_label_statistics.items()
    }


def statistics_fn(
    detection_onset_times, detection_offset_times,
    target_onset_times, target_offset_times,
    other_onset_times, other_offset_times,
    onset_collar, offset_collar, offset_collar_rate=0.,
    time_decimals=6,
):
    """

    Args:
        detection_onset_times (b,n):
        detection_offset_times:
        target_onset_times:
        target_offset_times:
        other_onset_times:
        other_offset_times:
        onset_collar:
        offset_collar:
        offset_collar_rate:
        time_decimals:

    Returns:

    """

    det_crit = detection_offset_times > detection_onset_times
    num_detections = det_crit.sum(-1)

    onset_dist = np.abs(detection_onset_times[..., None] - target_onset_times)
    onset_crit = np.round(
        onset_dist - onset_collar, decimals=time_decimals) <= 0.

    offset_collars = np.maximum(
        offset_collar,
        offset_collar_rate*(target_offset_times-target_onset_times),
    )
    offset_dist = np.abs(
        detection_offset_times[..., None] - target_offset_times)
    offset_crit = np.round(
        offset_dist - offset_collars, decimals=time_decimals) <= 0.
    hit_mat = det_crit[..., None] * onset_crit * offset_crit
    assert np.logical_or(hit_mat == 0, hit_mat == 1).all(), np.unique(hit_mat.flatten())
    invalid_detections = np.logical_or(
        np.any(hit_mat.sum(1) > 1, axis=1),
        np.any(hit_mat.sum(2) > 1, axis=1),
    )
    for idx in np.argwhere(invalid_detections).flatten():
        G = {}
        for det_idx, gt_idx in np.argwhere(hit_mat[idx]):
            if det_idx not in G:
                G[det_idx] = []
            G[det_idx].append(gt_idx)

        matching = sorted(bipartite_match(G).items())
        hit_mat[idx] = np.zeros_like(hit_mat[idx])
        for gt_idx, det_idx in matching:
            hit_mat[idx][det_idx, gt_idx] = 1
    tps = hit_mat.sum((1, 2))
    fps = num_detections - tps
    return {'tps': tps, 'fps': fps}
