import numpy as np
from functools import partial
from sed_scores_eval.utils import parallel
from sed_scores_eval.base_modules.bootstrap import bootstrap_from_deltas
from sed_scores_eval.base_modules.postprocessing import medfilt
from sed_scores_eval.intersection_based.intermediate_statistics import intermediate_statistics_deltas
from sed_scores_eval.intersection_based.psds import _unique_cummax_sort, psd_roc, psds_from_psd_roc, multi_class_psd_roc_from_single_class_psd_rocs


def psd_roc_postprocessing(
        scores, ground_truth, audio_durations, *, postprocessing_fn,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
        time_decimals=6, num_jobs=1,
):
    """Compute PSD-ROC after applying post-processing.

    Args:
        scores:
        ground_truth:
        audio_durations:
        postprocessing_fn:
        dtc_threshold:
        gtc_threshold:
        cttc_threshold:
        alpha_ct:
        alpha_st:
        unit_of_time:
        max_efpr:
        time_decimals:
        num_jobs:

    Returns:

    """
    scores = postprocessing_fn(scores, lazy=True)
    return psd_roc(
        scores=scores, ground_truth=ground_truth,
        audio_durations=audio_durations,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
        alpha_ct=alpha_ct, alpha_st=alpha_st,
        unit_of_time=unit_of_time, max_efpr=max_efpr,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )


def deltas_postprocessing(
        scores, ground_truth, postprocessing_fn, *,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        time_decimals=6, num_jobs=1,
):
    """Compute deltas after applying post-processing

    Args:
        scores:
        ground_truth:
        postprocessing_fn:
        dtc_threshold:
        gtc_threshold:
        cttc_threshold:
        time_decimals:
        num_jobs:

    Returns:

    """
    scores = postprocessing_fn(scores, lazy=True)
    return intermediate_statistics_deltas(
        scores=scores, ground_truth=ground_truth,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )


def merge_individual_rocs_into_overall_roc(rocs):
    """Combine operating points from multiple ROCS into a single RPC

    Args:
        rocs:

    Returns:

    """
    tprs, efprs, scores, filter_lengths = [
        np.concatenate(values) for values in list(zip(*rocs))
    ]
    return _unique_cummax_sort(tprs, efprs, scores, filter_lengths)


def postprocessing_independent_psds(
        scores, ground_truth, audio_durations, *, postprocessing_functions,
        scores_processed=None, deltas=None,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
        time_decimals=6, num_jobs=1,
):
    """
    Computes post-processing independent Polyphonic Sound Detection Score (piPSDS) [3],
    which is the normalized area under pi-PSD-ROC. Similar to PSD-ROC [1, 2] the
    pi-PSD-ROC is computed as the mean of single class pi-ROCs plus a penalty on
    across class std. Single class pi-ROCs combine operating points from a set
    of different post-processings, such that for each eFPR the pi-TPR(eFPR) is
    given as the max TPR over the set of post-processings.

    [1] C. Bilen, G. Ferroni, F. Tuveri, J. Azcarreta and S. Krstulovic,
    "A Framework for the Robust Evaluation of Sound Event Detection",
    in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
    2020, pp. 61–65

    [2] J.Ebbers, R.Serizel, and R.Haeb-Umbach
    "Threshold-Independent Evaluation of Sound Event Detection Scores",
    in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
    2022

    [3] J.Ebbers, R.Haeb-Umbach, and R.Serizel
    "Post-Processing Independent Evaluation of Sound Event Detection Systems",
    submitted to Detection and Classification of Acoustic Scenes and Events (DCASE) Workshop,
    2023

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        audio_durations: The duration of each audio file in the evaluation set.
        postprocessing_functions:
        scores_processed:
        deltas (dict of dicts of tuples): Must be deltas as returned by
            `accumulated_intermediate_statistics_from_deltas`. If not provided,
            deltas are computed within this function. Providing deltas is useful
            if deltas are used repeatedly as, e.g., with bootstrapped evaluation,
            to save computing time.
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
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, e.g., a detection with an ground truth intersection
            exactly matching the DTC, may be falsely counted as false detection
            because of small deviations due to limited floating point precision.
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns:
        psds (float): Polyphonic Sound Detection Score (PSDS), i.e., the area
            under the PSD ROC Curve up to max_efpr normalized by max_efpr.
        psd_roc (tuple of 1d np.ndarrays): tuple of effective True Positive
            Rates and effective False Positive Rates.
        single_class_psd_rocs (dict of tuples of 1d np.ndarrays):
            tuple of True Positive Rates and effective False Positive Rates
            for each event class.

    """
    if (
        isinstance(scores, (list, tuple))
        or (isinstance(scores_processed, (list, tuple)) and isinstance(scores_processed[0], (list, tuple)))
        or (isinstance(deltas, (list, tuple)) and isinstance(deltas[0], (list, tuple)))
    ):
        batch_size = [len(v) for v in [scores, scores_processed, deltas] if v is not None][0]
        return list(zip(*parallel.map(
            (scores, scores_processed, deltas), ('scores', 'scores_processed', 'deltas'),
            func=postprocessing_independent_psds, max_jobs=num_jobs,
            postprocessing_functions=postprocessing_functions,
            ground_truth=ground_truth, audio_durations=audio_durations,
            dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold, alpha_ct=alpha_ct,
            alpha_st=alpha_st, unit_of_time=unit_of_time, max_efpr=max_efpr,
            time_decimals=time_decimals,
            num_jobs=num_jobs//batch_size,
        )))
    (pi_effective_tp_rate, pi_effective_fp_rate), single_class_pi_psd_rocs, psd_rocs, single_class_psd_rocs = postprocessing_independent_psd_roc(
        scores=scores, ground_truth=ground_truth, audio_durations=audio_durations,
        postprocessing_functions=postprocessing_functions,
        scores_processed=scores_processed, deltas=deltas,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold, alpha_ct=alpha_ct, alpha_st=alpha_st,
        unit_of_time=unit_of_time, max_efpr=max_efpr,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )
    pi_psds_value = psds_from_psd_roc(
        pi_effective_tp_rate, pi_effective_fp_rate, max_efpr)
    single_class_pi_psds = {
        class_name: psds_from_psd_roc(tpr, efpr, max_efpr)
        for class_name, (tpr, efpr, *_) in single_class_pi_psd_rocs.items()
    }
    return (
        pi_psds_value,
        single_class_pi_psds,
        (pi_effective_tp_rate, pi_effective_fp_rate),
        single_class_pi_psd_rocs,
        psd_rocs,
        single_class_psd_rocs
    )


def postprocessing_independent_psd_roc(
        scores, ground_truth, audio_durations, *, postprocessing_functions,
        scores_processed=None, deltas=None,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
        time_decimals=6, num_jobs=1,
):
    """Similar to PSD-ROC [1,2] the pi-PSD-ROC [3] is computed as the mean of
    single class pi-ROCs plus a penalty on across class std. Single class
    pi-ROCs combine operating points from a set of different post-processings,
    such that for each eFPR the pi-TPR(eFPR) is given as the max TPR over the
    set of post-processings.

    [1] C. Bilen, G. Ferroni, F. Tuveri, J. Azcarreta and S. Krstulovic,
    "A Framework for the Robust Evaluation of Sound Event Detection",
    in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
    2020, pp. 61–65

    [2] J.Ebbers, R.Serizel, and R.Haeb-Umbach
    "Threshold-Independent Evaluation of Sound Event Detection Scores",
    in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
    2022

    [3] J.Ebbers, R.Haeb-Umbach, and R.Serizel
    "Post-Processing Independent Evaluation of Sound Event Detection Systems",
    submitted to Detection and Classification of Acoustic Scenes and Events (DCASE) Workshop,
    2023

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        audio_durations: The duration of each audio file in the evaluation set.
        postprocessing_functions:
        scores_processed:
        deltas (dict of dicts of tuples): Must be deltas as returned by
            `accumulated_intermediate_statistics_from_deltas`. If not provided,
            deltas are computed within this function. Providing deltas is useful
            if deltas are used repeatedly as, e.g., with bootstrapped evaluation,
            to save computing time.
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
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, e.g., a detection with an ground truth intersection
            exactly matching the DTC, may be falsely counted as false detection
            because of small deviations due to limited floating point precision.
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns:
        etpr (1d np.ndarray): effective MFI True Positive Rates.
        efpr (1d np.ndarray): effective False Positive Rates.
        single_class_psd_rocs (dict of tuples of 1d np.ndarrays):
            tuple of MFI True Positive Rates and effective False Positive Rates
            for each event class.

    """
    if (
        isinstance(scores, (list, tuple))
        or (isinstance(scores_processed, (list, tuple)) and isinstance(scores_processed[0], (list, tuple)))
        or (isinstance(deltas, (list, tuple)) and isinstance(deltas[0], (list, tuple)))
    ):
        # batch input
        batch_size = [len(v) for v in [scores, scores_processed, deltas] if v is not None][0]
        return list(zip(*parallel.map(
            (scores, scores_processed, deltas), ('scores', 'scores_processed', 'deltas'),
            func=postprocessing_independent_psd_roc, max_jobs=num_jobs,
            postprocessing_functions=postprocessing_functions,
            ground_truth=ground_truth, audio_durations=audio_durations,
            dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold, alpha_ct=alpha_ct,
            alpha_st=alpha_st, unit_of_time=unit_of_time, max_efpr=max_efpr,
            time_decimals=time_decimals,
            num_jobs=num_jobs//batch_size,
        )))
    if deltas is not None:
        assert isinstance(deltas, (list, tuple)), type(deltas)
        if scores_processed is None:
            scores_processed = len(deltas) * [None]
        if postprocessing_functions is not None:
            assert len(deltas) == len(postprocessing_functions), (len(deltas), len(postprocessing_functions))
    if scores_processed is not None:
        assert isinstance(scores_processed, (list, tuple)), type(scores_processed)
        if postprocessing_functions is not None:
            assert len(scores_processed) == len(postprocessing_functions), (len(scores_processed), len(postprocessing_functions))
        psd_rocs, single_class_psd_rocs = list(zip(*parallel.map(
            (scores_processed, deltas), arg_keys=('scores', 'deltas'),
            func=psd_roc, max_jobs=num_jobs,
            ground_truth=ground_truth, audio_durations=audio_durations,
            dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold, alpha_ct=alpha_ct,
            alpha_st=alpha_st, unit_of_time=unit_of_time, max_efpr=max_efpr,
            time_decimals=time_decimals,
            num_jobs=num_jobs//len(scores_processed),
        )))
    else:
        assert isinstance(postprocessing_functions, (list, tuple)), type(postprocessing_functions)
        psd_rocs, single_class_psd_rocs = list(zip(*parallel.map(
            postprocessing_functions, arg_keys='postprocessing_fn',
            func=psd_roc_postprocessing, max_jobs=num_jobs,
            scores=scores, ground_truth=ground_truth, audio_durations=audio_durations,
            dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold, alpha_ct=alpha_ct,
            alpha_st=alpha_st, unit_of_time=unit_of_time, max_efpr=max_efpr,
            time_decimals=time_decimals,
            num_jobs=1,
        )))

    single_class_pi_psd_rocs = {
        class_name: merge_individual_rocs_into_overall_roc([
            (
                *single_class_psd_rocs[m][class_name],
                np.full(
                    len(single_class_psd_rocs[m][class_name][-1]),
                    m,
                    dtype=int,
                )
            )
            for m in range(len(single_class_psd_rocs))
        ])
        for class_name in single_class_psd_rocs[0]
    }

    pi_psd_roc = multi_class_psd_roc_from_single_class_psd_rocs(
        single_class_pi_psd_rocs, alpha_st=alpha_st, max_efpr=max_efpr
    )
    return pi_psd_roc, single_class_pi_psd_rocs, psd_rocs, single_class_psd_rocs


def bootstrapped_postprocessing_independent_psds(
        scores, ground_truth, audio_durations, *, postprocessing_functions,
        scores_processed=None, deltas=None,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
        time_decimals=6, n_folds=5, n_iterations=4, num_jobs=1,
):
    """

    Args:
        scores:
        ground_truth:
        audio_durations:
        postprocessing_functions:
        scores_processed:
        deltas:
        dtc_threshold:
        gtc_threshold:
        cttc_threshold:
        alpha_ct:
        alpha_st:
        unit_of_time:
        max_efpr:
        time_decimals:
        n_folds:
        n_iterations:
        num_jobs:

    Returns:

    """
    if (
        isinstance(scores, (list, tuple))
        or (isinstance(scores_processed, (list, tuple)) and isinstance(scores_processed[0], (list, tuple)))
        or (isinstance(deltas, (list, tuple)) and isinstance(deltas[0], (list, tuple)))
    ):
        # batch input
        batch_size = [len(v) for v in [scores, scores_processed, deltas] if v is not None][0]
        return list(zip(*parallel.map(
            (scores, scores_processed, deltas),
            arg_keys=('scores', 'scores_processed', 'deltas'),
            func=bootstrapped_postprocessing_independent_psds,
            postprocessing_functions=postprocessing_functions,
            max_jobs=num_jobs, ground_truth=ground_truth,
            audio_durations=audio_durations,
            dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
            cttc_threshold=cttc_threshold, alpha_ct=alpha_ct,
            alpha_st=alpha_st, unit_of_time=unit_of_time, max_efpr=max_efpr,
            time_decimals=time_decimals,
            n_folds=n_folds, n_iterations=n_iterations,
            num_jobs=num_jobs//batch_size,
        )))
    if deltas is None:
        assert isinstance(postprocessing_functions, (list, tuple)), type(postprocessing_functions)
        if scores_processed is None:
            deltas = list(parallel.map(
                postprocessing_functions, arg_keys='postprocessing_fn',
                func=deltas_postprocessing, max_jobs=num_jobs,
                scores=scores, ground_truth=ground_truth,
                dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
                cttc_threshold=cttc_threshold, time_decimals=time_decimals,
                num_jobs=1,
            ))
        else:
            deltas = list(parallel.map(
                scores_processed, arg_keys='scores',
                func=intermediate_statistics_deltas, max_jobs=num_jobs,
                ground_truth=ground_truth,
                dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
                cttc_threshold=cttc_threshold, time_decimals=time_decimals,
                num_jobs=1,
            ))
    return bootstrap_from_deltas(
        postprocessing_independent_psds, deltas,
        n_folds=n_folds, n_iterations=n_iterations, num_jobs=num_jobs,
        scores=None, ground_truth=ground_truth, audio_durations=audio_durations,
        postprocessing_functions=postprocessing_functions,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold, alpha_ct=alpha_ct, alpha_st=alpha_st,
        unit_of_time=unit_of_time, max_efpr=max_efpr
    )


def median_filter_independent_psds(
        scores, ground_truth, audio_durations, *, median_filter_lengths_in_sec,
        scores_processed=None, deltas=None,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
        time_decimals=6, num_jobs=1,
):
    postprocessing_functions = [
        partial(medfilt, filter_length_in_sec=filter_length_in_sec)
        for filter_length_in_sec in median_filter_lengths_in_sec
    ]
    return postprocessing_independent_psds(
        scores, ground_truth, audio_durations,
        postprocessing_functions=postprocessing_functions,
        scores_processed=scores_processed, deltas=deltas,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold, alpha_ct=alpha_ct, alpha_st=alpha_st,
        unit_of_time=unit_of_time, max_efpr=max_efpr,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )


def bootstrapped_median_filter_independent_psds(
        scores, ground_truth, audio_durations, *, median_filter_lengths_in_sec,
        scores_processed=None, deltas=None,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
        time_decimals=6, n_folds=5, n_iterations=4, num_jobs=1,
):
    postprocessing_functions = [
        partial(medfilt, filter_length_in_sec=filter_length_in_sec)
        for filter_length_in_sec in median_filter_lengths_in_sec
    ]
    return bootstrapped_postprocessing_independent_psds(
        scores, ground_truth, audio_durations,
        postprocessing_functions=postprocessing_functions,
        scores_processed=scores_processed, deltas=deltas,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold, alpha_ct=alpha_ct, alpha_st=alpha_st,
        unit_of_time=unit_of_time, max_efpr=max_efpr,
        time_decimals=time_decimals, num_jobs=num_jobs,
        n_folds=n_folds, n_iterations=n_iterations,
    )
