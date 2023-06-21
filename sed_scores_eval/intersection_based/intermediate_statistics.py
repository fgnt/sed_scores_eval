import numpy as np
from sed_scores_eval.base_modules.scores import validate_score_dataframe
from sed_scores_eval.base_modules.ground_truth import event_counts_and_durations
from sed_scores_eval.base_modules import statistics
from sed_scores_eval.base_modules.io import parse_inputs, parse_ground_truth
from sed_scores_eval.base_modules.detection import onset_deltas


def intermediate_statistics_deltas(
        scores, ground_truth, *, dtc_threshold, gtc_threshold,
        cttc_threshold=None, time_decimals=6, num_jobs=1
):
    """Compute intersection-based intermediate statistics deltas for all audio
    files for all event classes and decision thresholds. See [1] for details
    about intersection-based evaluation. See [2] for details about the deltas
    and the joint computation of intermediate statistics for arbitrary decision
    thresholds.

    [1] C. Bilen, G. Ferroni, F. Tuveri, J. Azcarreta and S. Krstulovic,
    "A Framework for the Robust Evaluation of Sound Event Detection",
    in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
    2020, pp. 61–65

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
        dtc_threshold (float): detection tolerance criterion threshold
        gtc_threshold (float): ground truth intersection criterion threshold
        cttc_threshold (float): cross trigger tolerance criterion threshold
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, e.g., a detection with an ground truth intersection
            exactly matching the DTC, may be falsely counted as false detection
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
                 'tps' (1d np.ndarray): true positive counts for each score
                 'fps' (1d np.ndarray): false positive counts for each score
                 'cts' (dict of 1d np.ndarray): cross triggers with each of the
                    other classes for each score

    """
    scores, ground_truth, keys = parse_inputs(scores, ground_truth)
    _, event_classes = validate_score_dataframe(
        scores[keys[0]])
    return statistics.intermediate_statistics_deltas(
        scores, ground_truth,
        intermediate_statistics_fn=statistics_fn,
        acceleration_fn=acceleration_fn,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )


def accumulated_intermediate_statistics_from_deltas(deltas, ground_truth):
    """Compute intersection-based intermediate statistics over all audio files
    for all event classes and decision thresholds. See [1] for details about
    intersection-based evaluation. See [2] for details about the joint
    computation of intermediate statistics for arbitrary decision thresholds.

    [1] C. Bilen, G. Ferroni, F. Tuveri, J. Azcarreta and S. Krstulovic,
    "A Framework for the Robust Evaluation of Sound Event Detection",
    in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
    2020, pp. 61–65

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
                 'tps' (1d np.ndarray): true positive counts for each score
                 'fps' (1d np.ndarray): false positive counts for each score
                 'cts' (dict of 1d np.ndarray): cross triggers with each of the
                    other classes for each score
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.

    Returns (dict of tuples): for each event class a tuple of 1d scores array
        and a dict of intermediate statistics with the following keys
        (where each array has the same length as the scores array):
            "tps": true positives count array
            "fps": false positives count array
            "cts": dict of cross trigger count arrays with each other class
            "n_ref": integer number of target class ground truth events
            "t_ref": combined duration of all target class ground truth events
            "n_ref_other": dict of integer numbers of ground truth events from
                each other class
            "t_ref_other": dict of combined durations of ground truth events
                from each other class

    """
    multi_label_statistics = statistics.accumulated_intermediate_statistics_from_deltas(deltas)
    event_classes = multi_label_statistics.keys()
    n_ref, t_ref = event_counts_and_durations(
        ground_truth, event_classes=event_classes
    )
    return {
        class_name: (
            cp_scores_cls,
            {
                **stats_cls,
                'n_ref': n_ref[class_name],
                't_ref': t_ref[class_name],
                'n_ref_other': {
                    ocls: n_ref[ocls]
                    for ocls in event_classes if ocls != class_name
                },
                't_ref_other': {
                    ocls: t_ref[ocls]
                    for ocls in event_classes if ocls != class_name
                },
            },
        )
        for class_name, (cp_scores_cls, stats_cls) in multi_label_statistics.items()
    }


def accumulated_intermediate_statistics(
        scores, ground_truth, *, deltas=None, dtc_threshold, gtc_threshold,
        cttc_threshold=None, time_decimals=6, num_jobs=1
):
    """Cascade of `intermediate_statistics_deltas` and `accumulated_intermediate_statistics_from_deltas`.

    Args:
        scores:
        ground_truth:
        deltas:
        dtc_threshold:
        gtc_threshold:
        cttc_threshold:
        time_decimals:
        num_jobs:

    Returns:

    """
    if deltas is None:
        scores, ground_truth, audio_ids = parse_inputs(scores, ground_truth)
        deltas = intermediate_statistics_deltas(
            scores=scores, ground_truth=ground_truth,
            dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold,
            time_decimals=time_decimals, num_jobs=num_jobs,
        )
    else:
        audio_ids = list(deltas.keys())
        ground_truth = parse_ground_truth(
            ground_truth, audio_ids=audio_ids, additional_ids_ok=True)
    return accumulated_intermediate_statistics_from_deltas(deltas, ground_truth), audio_ids


def statistics_fn(
        detection_onset_times, detection_offset_times,
        target_onset_times, target_offset_times,
        other_onset_times, other_offset_times,
        dtc_threshold, gtc_threshold, cttc_threshold,
        time_decimals=6,
):
    """Compute intersection-based intermediate statistics for a single audio
    and single target class given detected onset/offset times, target class
    ground truth onset/offset times and other classes' ground truth
    onset/offset times.

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
        other_onset_times (dict of 1d np.ndarrays): onset times of other class
            ground truth events
        other_offset_times (dict of 1d np.ndarrays): offset times of other
            class ground truth events
        dtc_threshold (float): detection tolerance criterion threshold
        gtc_threshold (float): ground truth intersection criterion threshold
        cttc_threshold (float): cross trigger tolerance criterion threshold
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, e.g., a detection with a ground truth intersection
            exactly matching the DTC, may be falsely counted as false detection
            because of small deviations due to limited floating point precision.

    Returns (dict of 1d np.ndarrays): dict of intermediate statistics with the
        following keys (where each array has the length B):
            "tps": true positives count array
            "fps": false positives count array
            "cts": dict of cross trigger count arrays with each other class

    """
    det_crit = detection_offset_times > detection_onset_times
    num_detections = det_crit.sum(-1)
    ground_truth_intersections = np.maximum(
        np.minimum(detection_offset_times[..., None], target_offset_times)
        - np.maximum(detection_onset_times[..., None], target_onset_times),
        0.,
    )
    total_intersection_with_gt_events = np.round(
        np.sum(ground_truth_intersections, axis=-1), decimals=time_decimals)
    detection_lengths = detection_offset_times - detection_onset_times
    dtc = (
        total_intersection_with_gt_events
        >= np.round(dtc_threshold * detection_lengths, decimals=time_decimals)
    ) * det_crit
    num_relevant_detections = dtc.sum(-1)
    fps = num_detections - num_relevant_detections
    total_intersection_with_relevant_detections = np.round(
        np.sum(dtc[..., None] * ground_truth_intersections, axis=-2),
        decimals=time_decimals
    )
    gt_lengths = target_offset_times - target_onset_times
    gtc = (
        total_intersection_with_relevant_detections
        >= np.round(gtc_threshold * gt_lengths, decimals=time_decimals)
    )
    tps = gtc.sum(-1)
    if cttc_threshold is None:
        cts = np.zeros_like(tps)
    else:
        cts = {}
        assert other_onset_times.keys() == other_offset_times.keys(), (other_onset_times.keys(), other_offset_times.keys())
        for other_class_name in other_onset_times.keys():
            gt_onset_times = other_onset_times[other_class_name]
            gt_offset_times = other_offset_times[other_class_name]
            assert len(gt_onset_times) == len(gt_offset_times)
            if len(gt_onset_times) == 0:
                cts[other_class_name] = np.zeros_like(tps)
            else:
                other_class_intersections = np.maximum(
                    np.minimum(detection_offset_times[..., None], gt_offset_times)
                    - np.maximum(detection_onset_times[..., None], gt_onset_times),
                    0.,
                )
                total_intersection_with_other_gt_events = np.round(
                    np.sum((1-dtc[..., None]) * other_class_intersections, axis=-1),
                    decimals=time_decimals
                )
                cttc = (
                    total_intersection_with_other_gt_events
                    >= np.round(cttc_threshold * detection_lengths, decimals=time_decimals)
                ) * det_crit
                cts[other_class_name] = cttc.sum(-1)
    return {'tps': tps, 'fps': fps, 'cts': cts}


def acceleration_fn(
        scores, timestamps,
        target_onset_times, target_offset_times,
        other_onset_times, other_offset_times,
        dtc_threshold, gtc_threshold, cttc_threshold,
        time_decimals=6,
):
    # return np.unique(scores), None, None
    onset_deltas_ = onset_deltas(scores)
    change_points = np.abs(onset_deltas_) > .5
    if (len(target_onset_times) == 0) and (
        (cttc_threshold is None) or all([
            len(gt_onset_times) == 0 for gt_onset_times in other_onset_times
        ])
    ):
        cp_scores = scores[change_points]
        if cttc_threshold is None:
            cts_deltas = np.zeros_like(cp_scores)
        else:
            cts_deltas = {
                other_class_name: np.zeros_like(cp_scores)
                for other_class_name in other_onset_times.keys()
            }
        deltas = {
            'fps': onset_deltas_[change_points],
            'tps': np.zeros_like(cp_scores),
            'cts': cts_deltas,
        }
        return None, cp_scores, deltas
    return None, None, None
    # if cttc_threshold is None:
    #     gt_onset_times = target_onset_times
    #     gt_offset_times = target_offset_times
    # else:
    #     other_class_names = list(other_onset_times.keys())
    #     gt_onset_times = np.concatenate([target_onset_times] + [other_onset_times[ocls] for ocls in other_class_names])
    #     gt_offset_times = np.concatenate([target_offset_times] + [other_offset_times[ocls] for ocls in other_class_names])
    #
    # for onset_time, offset_time in zip(gt_onset_times, gt_offset_times):
    #     onset_idx, offset_idx = onset_offset_times_to_score_indices(onset_time, offset_time, timestamps)
    #     change_points[onset_idx:offset_idx] = True
    #     right_sided_cummin_indices = offset_idx-1 + cummin(scores[offset_idx-1:])[1]
    #     change_points[right_sided_cummin_indices] = True
    #     left_sided_cummin_indices = onset_idx - cummin(scores[:onset_idx+1][::-1])[1]
    #     change_points[left_sided_cummin_indices] = True
    #
    # return np.unique(scores[change_points]), None, None
