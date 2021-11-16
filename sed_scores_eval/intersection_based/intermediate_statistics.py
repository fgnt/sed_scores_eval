import numpy as np
from sed_scores_eval.utils.scores import extract_timestamps_and_classes_from_dataframe
from sed_scores_eval.base_modules.ground_truth import event_counts_and_durations
from sed_scores_eval.base_modules.statistics import accumulated_intermediate_statistics
from sed_scores_eval.base_modules.io import parse_inputs


def intermediate_statistics(
        scores, ground_truth, dtc_threshold, gtc_threshold,
        cttc_threshold=None, time_decimals=6,
):
    """

    Args:
        scores (dict of pandas.DataFrames): score DataFrames for each audio
            clip of a data set. Each DataFrame contains onset and offset times
            of a score window  in first two columns followed by sed score
            columns for each event class.
        ground_truth (dict of lists of tuples): list of ground truth event
            tuples (onset, offset, event class) for each audio clip.
        dtc_threshold (float): detection tolerance criterion threshold
        gtc_threshold (float): ground truth intersection criterion threshold
        cttc_threshold (float): cross trigger tolerance criterion threshold
        time_decimals:

    Returns:

    """
    scores, ground_truth, keys = parse_inputs(scores, ground_truth)
    _, event_classes = extract_timestamps_and_classes_from_dataframe(
        scores[keys[0]])
    multi_label_statistics = accumulated_intermediate_statistics(
        scores, ground_truth,
        intermediate_statistics_fn=statistics_fn,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
        time_decimals=time_decimals,
    )
    n_ref, t_ref = event_counts_and_durations(
        ground_truth, event_classes=multi_label_statistics.keys()
    )
    return {
        cls: (
            cp_scores_cls,
            {
                **stats_cls,
                'n_ref': n_ref[cls],
                't_ref': t_ref[cls],
                'n_ref_other': [
                    n_ref[ocls] for ocls in event_classes if ocls != cls],
                't_ref_other': [
                    t_ref[ocls] for ocls in event_classes if ocls != cls],
            },
        )
        for cls, (cp_scores_cls, stats_cls) in multi_label_statistics.items()
    }


def statistics_fn(
        detection_onset_times, detection_offset_times,
        target_onset_times, target_offset_times,
        other_onset_times, other_offset_times,
        dtc_threshold, gtc_threshold, cttc_threshold,
        time_decimals=6,
):
    """

    Args:
        detection_onset_times:
        detection_offset_times:
        target_onset_times:
        target_offset_times:
        other_onset_times:
        other_offset_times:
        dtc_threshold:
        gtc_threshold:
        cttc_threshold:
        time_decimals:

    Returns:

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
