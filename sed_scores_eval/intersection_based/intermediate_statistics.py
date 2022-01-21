import numpy as np
from sed_scores_eval.utils.scores import validate_score_dataframe
from sed_scores_eval.base_modules.ground_truth import event_counts_and_durations
from sed_scores_eval.base_modules.statistics import accumulated_intermediate_statistics
from sed_scores_eval.base_modules.io import parse_inputs


def intermediate_statistics(
        scores, ground_truth, dtc_threshold, gtc_threshold,
        cttc_threshold=None, time_decimals=6, num_jobs=1
):
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
    submitted to IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
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

    Returns (dict of tuples): for each event class a tuple of 1d scores array
        and a dict of intermediate statistics with the following keys
        (where each array has the same length as the scores array):
            "tps": true positives count array
            "fps": false positives count array
            "cts": list of cross trigger count arrays with each other class
            "n_ref": integer number of target class ground truth events
            "t_ref": combined duration of all target class ground truth events
            "n_ref_other": list of integer numbers of ground truth events from
                each other class
            "t_ref_other": list of combined durations of ground truth events
                from each other class

    """
    scores, ground_truth, keys = parse_inputs(scores, ground_truth)
    _, event_classes = validate_score_dataframe(
        scores[keys[0]])
    multi_label_statistics = accumulated_intermediate_statistics(
        scores, ground_truth,
        intermediate_statistics_fn=statistics_fn,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )
    n_ref, t_ref = event_counts_and_durations(
        ground_truth, event_classes=multi_label_statistics.keys()
    )
    return {
        class_name: (
            cp_scores_cls,
            {
                **stats_cls,
                'n_ref': n_ref[class_name],
                't_ref': t_ref[class_name],
                'n_ref_other': [
                    n_ref[ocls] for ocls in event_classes if ocls != class_name],
                't_ref_other': [
                    t_ref[ocls] for ocls in event_classes if ocls != class_name],
            },
        )
        for class_name, (cp_scores_cls, stats_cls) in multi_label_statistics.items()
    }


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
        other_onset_times (list of 1d np.ndarrays): onset times of other class
            ground truth events
        other_offset_times (list of 1d np.ndarrays): offset times of other
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
            "cts": list of cross trigger count arrays with each other class

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
    detection_lengths = np.round(
        detection_offset_times-detection_onset_times, decimals=time_decimals)
    detection_lengths[detection_lengths == 0.] = 1e-12
    dtc_scores = total_intersection_with_gt_events / detection_lengths
    dtc = dtc_scores >= dtc_threshold
    num_relevant_detections = dtc.sum(-1)
    fps = num_detections - num_relevant_detections
    total_intersection_with_relevant_detections = np.round(
        np.sum(dtc[..., None] * ground_truth_intersections, axis=-2),
        decimals=time_decimals
    )
    gt_lengths = np.round(
        target_offset_times-target_onset_times, decimals=time_decimals)
    gtc_scores = total_intersection_with_relevant_detections / gt_lengths
    gtc = gtc_scores >= gtc_threshold
    tps = gtc.sum(-1)
    if cttc_threshold is None:
        cts = np.zeros_like(tps)
    else:
        cts = []
        for gt_onset_times, gt_offset_times in zip(other_onset_times, other_offset_times):
            other_class_intersections = np.maximum(
                np.minimum(detection_offset_times[..., None], gt_offset_times)
                - np.maximum(detection_onset_times[..., None], gt_onset_times),
                0.,
            )
            total_intersection_with_other_gt_events = np.round(
                np.sum((1-dtc[..., None])*other_class_intersections, axis=-1),
                decimals=time_decimals
            )
            cttc_scores = (
                total_intersection_with_other_gt_events / detection_lengths
            )
            cttc = cttc_scores >= cttc_threshold
            cts.append(cttc.sum(-1))
        cts = np.array(cts).T
    return {'tps': tps, 'fps': fps, 'cts': cts}
