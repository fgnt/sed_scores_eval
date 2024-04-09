import numpy as np
import pandas as pd
from pathlib import Path
from sed_scores_eval.utils.array_ops import get_first_index_where
from sed_scores_eval.base_modules.scores import validate_score_dataframe
from sed_scores_eval.base_modules.detection import scores_to_event_list
from sed_scores_eval.base_modules.io import (
    parse_inputs, write_detections_for_multiple_thresholds
)
from sed_scores_eval.intersection_based.psds import psds_from_psd_roc


def approximate_psds(
        scores, ground_truth, audio_durations,
        thresholds=np.linspace(.01, .99, 50), *,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
        score_transform=None,
):
    """Reference psds implementation using the psds_eval package
    (https://github.com/audioanalytic/psds_eval), which, however, only
    approximates the PSD-ROC using a limited set of thresholds/operating points.
    This function is primarily used for testing purposes.

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        audio_durations: The duration of each audio file in the evaluation set.
        thresholds: the set of thresholds used to approximate the PSD-ROC.
        dtc_threshold (float): detection tolerance criterion threshold
        gtc_threshold (float): ground truth intersection criterion threshold
        cttc_threshold (float): cross trigger tolerance criterion threshold
        alpha_ct (float): parameter for penalizing cross triggers.
            More specifically, it is the weight of the cross trigger rate
            (averaged over all other classes) that is added to the False
            Positive Rate (FPR) yielding the effective FPR (eFPR). Default is 0.
        alpha_st (float): parameter for penalizing instability across classes.
            More specifically, it is the weight of the standard deviation of
            the per-class ROCs, that is subtracted from the mean of the
            per-class ROCs. Default is 0.
        unit_of_time (str): the unit of time \in {second, minute, hour} to be
            used for computation of the eFPR (which is defined as rate per unit
            of time). Default is hour.
        max_efpr (float): the maximum eFPR for which the system is evaluated,
            i.e., until where the Area under PSD ROC Curve is computed.
            Default is 100.
        score_transform: a (non-linear) score transformation may be used before
            thresholding to obtain a better PSD-ROC approximation [1].
            [1] J.Ebbers, R.Serizel, and R.Haeb-Umbach
            "Threshold-Independent Evaluation of Sound Event Detection Scores",
            in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
            2022

    Returns:
        psds (float): Polyphonic Sound Detection Score (PSDS), i.e., the area
            under the approximated PSD ROC Curve up to max_efpr normalized
            by max_efpr.
        psd_roc (tuple of 1d np.ndarrays): tuple of effective True Positive
            Rates and effective False Positive Rates.
        single_class_psd_rocs (dict of tuples of 1d np.ndarrays):
            tuple of True Positive Rates and effective False Positive Rates
            for each event class.

    """
    import tempfile
    scores, _, keys = parse_inputs(scores, ground_truth)
    ground_truth = _parse_events(ground_truth)
    audio_format = ground_truth['filename'][0].split('.')[-1]
    validate_score_dataframe(scores[keys[0]])
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir)
        write_detections_for_multiple_thresholds(
            scores, thresholds, dir_path, audio_format=audio_format,
            score_transform=score_transform, threshold_decimals=16,
        )
        return approximate_psds_from_detections_dir(
            dir_path, ground_truth, audio_durations, thresholds,
            dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold,
            alpha_ct=alpha_ct, alpha_st=alpha_st,
            unit_of_time=unit_of_time, max_efpr=max_efpr,
            threshold_decimals=16,
        )


def approximate_psds_from_detections_dir(
        dir_path, ground_truth, audio_durations,
        thresholds=np.linspace(.01, .99, 50), *,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
        threshold_decimals=3,
):
    """Reference psds implementation using the psds_eval package
    (https://github.com/audioanalytic/psds_eval), which, however, only
    approximates the PSD-ROC using a limited set of thresholds/operating points.
    This function is primarily used for testing purposes.

    Args:
        dir_path (str, pathlib.Path): path to directory where the detections
            for each threshold are saved.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        audio_durations: The duration of each audio file in the evaluation set.
        thresholds: the set of thresholds used to approximate the PSD-ROC.
        dtc_threshold (float): detection tolerance criterion threshold
        gtc_threshold (float): ground truth intersection criterion threshold
        cttc_threshold (float): cross trigger tolerance criterion threshold
        alpha_ct (float): parameter for penalizing cross triggers.
            More specifically, it is the weight of the cross trigger rate
            (averaged over all other classes) that is added to the False
            Positive Rate (FPR) yielding the effective FPR (eFPR). Default is 0.
        alpha_st (float): parameter for penalizing instability across classes.
            More specifically, it is the weight of the standard deviation of
            the per-class ROCs, that is subtracted from the mean of the
            per-class ROCs. Default is 0.
        unit_of_time (str): the unit of time \in {second, minute, hour} to be
            used for computation of the eFPR (which is defined as rate per unit
            of time). Default is hour.
        max_efpr (float): the maximum eFPR for which the system is evaluated,
            i.e., until where the Area under PSD ROC Curve is computed.
            Default is 100.
        score_transform: a (non-linear) score transformation may be used before
            thresholding to obtain a better PSD-ROC approximation [1].
            [1] J.Ebbers, R.Serizel, and R.Haeb-Umbach
            "Threshold-Independent Evaluation of Sound Event Detection Scores",
            in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
            2022
        threshold_decimals:

    Returns:
        psds (float): Polyphonic Sound Detection Score (PSDS), i.e., the area
            under the approximated PSD ROC Curve up to max_efpr normalized
            by max_efpr.
        psd_roc (tuple of 1d np.ndarrays): tuple of effective True Positive
            Rates and effective False Positive Rates.
        single_class_psd_rocs (dict of tuples of 1d np.ndarrays):
            tuple of True Positive Rates and effective False Positive Rates
            for each event class.

    """
    assert np.all(np.abs(thresholds - np.round(thresholds, threshold_decimals)) < 1e-15), (threshold_decimals, thresholds)
    assert np.all(thresholds == np.unique(thresholds)), thresholds
    from psds_eval import PSDSEval
    ground_truth = _parse_events(ground_truth)
    audio_durations = _parse_audio_durations(audio_durations)
    dir_path = Path(dir_path)
    psds_eval = PSDSEval(
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=1. if cttc_threshold is None else cttc_threshold,
        ground_truth=ground_truth,
        metadata=audio_durations,
        duration_unit=unit_of_time,
    )
    psds_eval.clear_all_operating_points()

    for i, threshold in enumerate(thresholds):
        tsv = dir_path / '{:.Xf}.tsv'.replace('X', str(threshold_decimals)).format(threshold)
        print(f"Adding operating point {i+1}/{len(thresholds)}", end="\r")
        det = pd.read_csv(tsv, sep="\t")
        info = {"name": f"Op {i+1:02d}", "threshold": threshold}
        psds_eval.add_operating_point(det, info=info)

    # compute the PSDS of the system represented by its operating points
    psds_ = psds_eval.psds(
        max_efpr=max_efpr,
        alpha_st=alpha_st,
        alpha_ct=alpha_ct,
    )
    efpr = psds_.plt.xp
    etpr = psds_.plt.yp
    _, _, tpr_vs_efpr = psds_eval.psd_roc_curves(alpha_ct=alpha_ct)

    single_class_psd_rocs = {
        psds_eval.class_names[i]: (tpr_vs_efpr.yp[i], tpr_vs_efpr.xp)
        for i in range(len(tpr_vs_efpr.yp))
    }

    if max_efpr is not None:
        cutoff_idx = get_first_index_where(efpr, "gt", max_efpr)
        etpr = np.array(etpr[:cutoff_idx].tolist() + [etpr[cutoff_idx-1]])
        efpr = np.array(efpr[:cutoff_idx].tolist() + [max_efpr])
        for key, roc in single_class_psd_rocs.items():
            cutoff_idx = get_first_index_where(roc[1], "gt", max_efpr)
            single_class_psd_rocs[key] = (
                roc[0][:cutoff_idx], roc[1][:cutoff_idx]
            )

    single_class_psds = {
        class_name: psds_from_psd_roc(tpr, efpr, max_efpr)
        for class_name, (tpr, efpr) in single_class_psd_rocs.items()
    }
    return psds_.value, single_class_psds, (etpr, efpr), single_class_psd_rocs


def fscore(scores, ground_truth, threshold, *, dtc_threshold, gtc_threshold, beta=1.):

    from psds_eval import PSDSEval

    scores, ground_truth, audio_ids = parse_inputs(scores, ground_truth)
    detections = _parse_events(scores_to_event_list(scores, threshold))
    gt = _parse_events(ground_truth)
    audio_durations = _parse_audio_durations({clip_id: scores_df['offset'].to_numpy()[-1] for clip_id, scores_df in scores.items()})
    psds = PSDSEval(
        ground_truth=gt,
        metadata=audio_durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
    )
    if detections.empty:
        _, event_classes = validate_score_dataframe(scores[audio_ids[0]])
        f_macro = .0
        f = {event_class: .0 for event_class in event_classes}
    else:
        f_macro, f = psds.compute_macro_f_score(detections, beta=beta)
    for event_class in f:
        if np.isnan(f[event_class]):
            f[event_class] = 0.0
    f['macro_average'] = f_macro
    return f


def _parse_events(events):
    if isinstance(events, pd.DataFrame):
        return events
    elif isinstance(events, dict):
        events = [(f"{key}.wav", *event) for key in sorted(events.keys()) for event in events[key]]
        events = pd.DataFrame(
            events,
            columns=['filename', 'onset', 'offset', 'event_label']
        )
    else:
        assert isinstance(events, (str, Path)), type(events)
        events = pd.read_csv(events, sep="\t")
    return events


def _parse_audio_durations(audio_durations):
    if isinstance(audio_durations, pd.DataFrame):
        return audio_durations
    elif isinstance(audio_durations, dict):
        audio_durations = [(f"{key}.wav", audio_durations[key]) for key in audio_durations.keys()]
        audio_durations = pd.DataFrame(
            audio_durations, columns=['filename', 'duration']
        )
    else:
        assert isinstance(audio_durations, (str, Path)), type(audio_durations)
        audio_durations = pd.read_csv(audio_durations, sep="\t")
    return audio_durations
