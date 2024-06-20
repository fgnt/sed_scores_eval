import numpy as np
from sed_eval.util import bipartite_match
from sed_scores_eval.base_modules.scores import validate_score_dataframe
from sed_scores_eval.base_modules.ground_truth import event_counts_and_durations
from sed_scores_eval.base_modules import statistics
from sed_scores_eval.base_modules.io import parse_inputs, parse_ground_truth
from sed_scores_eval.base_modules.detection import onset_deltas


def intermediate_statistics_deltas(
    scores, ground_truth, onset_collar, offset_collar, offset_collar_rate=0.,
    return_onset_offset_dist_sum=False, time_decimals=6, num_jobs=1,
):
    """Compute collar-based intermediate statistics deltas for all audio files
    for all event classes and decision thresholds. See [1] for details about
    collar-based (event-based) evaluation. See [2] for details about the deltas
    and the joint computation of intermediate statistics for arbitrary decision
    thresholds.

    [1] Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen,
    "Metrics for polyphonic sound event detection",
    Applied Sciences, vol. 6, pp. 162, 2016

    [2] J.Ebbers, R.Serizel, and R.Haeb-Umbach
    "Threshold-Independent Evaluation of Sound Event Detection Scores",
    in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
    2022

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        onset_collar (float): allowed onset deviation in seconds
        offset_collar (float): (at least) allowed offset deviation in seconds
        offset_collar_rate (float): (at least) allowed offset deviation as a
            ratio of the length of the ground truth event, with the actual
            allowed offset deviation being:
            offset_collar_for_gt_event = max(
                offset_collar, offset_collar_rate*length_of_gt_event_in_seconds
            )
        return_onset_offset_dist_sum: If True, return summed distances between
            predicted and true on-/offsets (for true positive predictions),
            which allows to compute and compensate biases.
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, detections with an onset or offset right on the
            boundary of the collar may be falsely counted as false detection
            because of small deviations due to limited floating point precision.
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns:
        deltas (dict of dicts of tuples): For each audio clip for each event class:
            change_point_scores (dict of dicts): 1d array of scores at which the
                intermediate statistics change, when the threshold falls below
                it, for that particular clip and class.
            dict of delta values: provides for each intermediate statistic a
                1d array of the delta (change), when the threshold falls below
                each of the change_point_scores.
                Required intermediate statistics deltas:
                  "tps": true positives count array
                  "fps": false positives count array

    """
    scores, ground_truth, keys = parse_inputs(scores, ground_truth)
    _, event_classes = validate_score_dataframe(
        scores[keys[0]])
    return statistics.intermediate_statistics_deltas(
        scores, ground_truth,
        intermediate_statistics_fn=statistics_fn,
        acceleration_fn=acceleration_fn,
        onset_collar=onset_collar, offset_collar=offset_collar,
        offset_collar_rate=offset_collar_rate,
        return_onset_offset_dist_sum=return_onset_offset_dist_sum,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )


def accumulated_intermediate_statistics_from_deltas(deltas, ground_truth):
    """Compute collar-based intermediate statistics over all audio files
    for all event classes and decision thresholds. See [1] for details about
    collar-based (event-based) evaluation. See [2] for details about the joint
    computation of intermediate statistics for arbitrary decision thresholds.

    [1] Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen,
    "Metrics for polyphonic sound event detection",
    Applied Sciences, vol. 6, pp. 162, 2016

    [2] J.Ebbers, R.Serizel, and R.Haeb-Umbach
    "Threshold-Independent Evaluation of Sound Event Detection Scores",
    in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
    2022

    Args:
        deltas (dict of dicts of tuples): For each audio clip for each event class:
            change_point_scores (dict of dicts): 1d array of scores at which the
                intermediate statistics change, when the threshold falls below
                it, for that particular clip and class.
            dict of delta values: provides for each intermediate statistic a
                1d array of the delta (change), when the threshold falls below
                each of the change_point_scores.
                Required intermediate statistics deltas:
                  "tps": true positives count array
                  "fps": false positives count array
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.

    Returns (dict of tuples): for each event class a tuple of 1d scores array
        and a dict of intermediate statistics with the following keys
        (where each array has the same length as the scores array):
            "tps": true positives count array
            "fps": false positives count array
            "n_ref": integer number of ground truth events

    """
    multi_label_statistics = statistics.accumulated_intermediate_statistics_from_deltas(deltas)
    n_ref, _ = event_counts_and_durations(
        ground_truth, event_classes=multi_label_statistics.keys()
    )
    return {
        class_name: (cp_scores_cls, {**stats_cls, 'n_ref': n_ref[class_name]})
        for class_name, (cp_scores_cls, stats_cls) in multi_label_statistics.items()
    }


def accumulated_intermediate_statistics(
    scores, ground_truth, *, deltas=None,
    onset_collar, offset_collar, offset_collar_rate=0.,
    return_onset_offset_dist_sum=False, time_decimals=6, num_jobs=1,
):
    """Cascade of `intermediate_statistics_deltas` and `accumulated_intermediate_statistics_from_deltas`.

    Args:
        scores:
        ground_truth:
        deltas:
        onset_collar:
        offset_collar:
        offset_collar_rate:
        return_onset_offset_dist_sum:
        time_decimals:
        num_jobs:

    Returns:

    """
    if deltas is None:
        scores, ground_truth, audio_ids = parse_inputs(scores, ground_truth)
        deltas = intermediate_statistics_deltas(
            scores, ground_truth,
            onset_collar=onset_collar, offset_collar=offset_collar,
            offset_collar_rate=offset_collar_rate,
            return_onset_offset_dist_sum=return_onset_offset_dist_sum,
            time_decimals=time_decimals, num_jobs=num_jobs,
        )
    else:
        audio_ids = list(deltas.keys())
        ground_truth = parse_ground_truth(ground_truth, audio_ids=audio_ids)
    return accumulated_intermediate_statistics_from_deltas(deltas, ground_truth), audio_ids


def statistics_fn(
    detection_onset_times, detection_offset_times,
    target_onset_times, target_offset_times,
    other_onset_times, other_offset_times,
    onset_collar, offset_collar, offset_collar_rate=0.,
    return_onset_offset_dist_sum=False,
    time_decimals=6,
):
    """Compute collar-based intermediate statistics for a single audio and
    single target class given detected onset/offset times and target class
    ground truth onset/offset times

    Args:
        detection_onset_times (np.ndarray): (B, M) onset times of detected
            target class events with M being the number of detected target
            class events, and B being an independent dimension.
        detection_offset_times (np.ndarray): (B, M) offset times of detected
            target class events with M being the number of detected target
            class events, and B being an independent dimension. Note that it
            may include offset times which are equal to the corresponding onset
            time, which indicates that the event is inactive at that specific
            position b along the independent axis and must not be counted as a
            detection.
        target_onset_times (1d np.ndarray): onset times of target class ground
            truth events.
        target_offset_times (1d np.ndarray): offset times of target class
            ground truth events.
        other_onset_times (list of 1d np.ndarrays): onset times of other class
            ground truth events
        other_offset_times (list of 1d np.ndarrays): offset times of other
            class ground truth events
        onset_collar (float): allowed onset deviation in seconds
        offset_collar (float): (at least) allowed offset deviation in seconds
        offset_collar_rate (float): (at least) allowed offset deviation as a
            ratio of the length of the ground truth event, with the actual
            allowed offset deviation being:
            offset_collar_for_gt_event = max(
                offset_collar, offset_collar_rate*length_of_gt_event_in_seconds
            )
        return_onset_offset_dist_sum: If True, return summed distances between
            predicted and true on-/offsets (for true positive predictions),
            which allows to compute and compensate biases.
            #ToDo: add on-/offset bias compensation
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, detections with an onset or offset right on the
            boundary of the collar may be falsely counted as false detection
            because of small deviations due to limited floating point precision.

    Returns (dict of 1d np.ndarrays): dict of intermediate statistics with the
        following keys (where each array has the length B):
            "tps": true positives count array
            "fps": false positives count array

    """

    det_crit = detection_offset_times > detection_onset_times
    num_detections = det_crit.sum(-1)

    onset_dist = detection_onset_times[..., None] - target_onset_times
    onset_crit = np.round(
        np.abs(onset_dist) - onset_collar, decimals=time_decimals) <= 0.

    offset_collars = np.maximum(
        offset_collar,
        offset_collar_rate * (target_offset_times-target_onset_times),
    )
    offset_dist = detection_offset_times[..., None] - target_offset_times
    offset_crit = np.round(
        np.abs(offset_dist) - offset_collars, decimals=time_decimals) <= 0.
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
    stats = {
        'tps': tps,
        'fps': fps,
    }
    if return_onset_offset_dist_sum:
        onset_dist = (hit_mat * onset_dist).sum((1, 2))
        offset_dist = (hit_mat * offset_dist).sum((1, 2))
        stats.update({
            'onset_dist_sum': onset_dist,
            'offset_dist_sum': offset_dist,
        })
    return stats


def acceleration_fn(
        scores, timestamps,
        target_onset_times, target_offset_times,
        other_onset_times, other_offset_times,
        onset_collar, offset_collar, offset_collar_rate=0.,
        return_onset_offset_dist_sum=False,
        time_decimals=6,
):
    # return np.unique(scores), None, None
    onset_deltas_ = onset_deltas(scores)
    change_points = np.abs(onset_deltas_) > .5
    if len(target_onset_times) == 0:
        cp_scores = scores[change_points]
        deltas = {
            'fps': onset_deltas_[change_points],
            'tps': np.zeros_like(cp_scores),
        }
        if return_onset_offset_dist_sum:
            deltas.update({
                'onset_dist_sum': np.zeros_like(cp_scores),
                'offset_dist_sum': np.zeros_like(cp_scores),
            })
        return None, cp_scores, deltas
    return None, None, None
