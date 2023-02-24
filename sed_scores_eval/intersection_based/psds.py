import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from sed_scores_eval.utils.array_ops import cummax, get_first_index_where
from sed_scores_eval.base_modules.io import parse_inputs, read_audio_durations
from sed_scores_eval.utils.auc import staircase_auc
from sed_scores_eval.intersection_based.intermediate_statistics import intermediate_statistics

seconds_per_unit_of_time = {
    'second': 1.,
    'minute': 60.,
    'hour': 3600.,
}


def psds(
        scores, ground_truth, audio_durations, *,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
        time_decimals=6, num_jobs=1,
):
    """Computes Polyphonic Sound Detection Score (PSDS) [1] using the exact
    and efficient computation approach proposed in [2].

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
        audio_durations: The duration of each audio file in the evaluation set.
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
    effective_tp_rate, effective_fp_rate, single_class_psds_rocs = psd_roc(
        scores=scores, ground_truth=ground_truth, audio_durations=audio_durations,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold, alpha_ct=alpha_ct, alpha_st=alpha_st,
        unit_of_time=unit_of_time, max_efpr=max_efpr,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )
    psd_roc_auc = staircase_auc(
        effective_tp_rate, effective_fp_rate, max_x=max_efpr)
    return (
        psd_roc_auc/max_efpr,
        (effective_tp_rate, effective_fp_rate),
        single_class_psds_rocs
    )


def psd_roc(
        scores, ground_truth, audio_durations, *,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
        time_decimals=6, num_jobs=1,
):
    """Computes Polyphonic Sound Detection ROC (PSD ROC) [1] using the exact
    and efficient computation approach proposed in [2].

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
        audio_durations: The duration of each audio file in the evaluation set.
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
        etpr (1d np.ndarray): effective True Positive Rates.
        efpr (1d np.ndarray): effective False Positive Rates.
        single_class_psd_rocs (dict of tuples of 1d np.ndarrays):
            tuple of True Positive Rates and effective False Positive Rates
            for each event class.

    """
    if alpha_ct == 0.:
        if cttc_threshold is not None:
            raise ValueError(
                'cttc_threshold has been provided but alpha_ct is 0.'
            )
    else:
        if cttc_threshold is None:
            raise ValueError(
                'alpha_ct is not 0 but no cttc_threshold has been provided.'
            )

    scores, ground_truth, audio_ids = parse_inputs(scores, ground_truth)
    if not isinstance(audio_durations, (dict, str, Path)):
        raise ValueError(
            f'audio_durations must be dict, str or Path but '
            f'{type(audio_durations)} was given.'
        )
    if isinstance(audio_durations, (str, Path)):
        audio_durations = Path(audio_durations)
        assert audio_durations.is_file(), audio_durations
        audio_durations = read_audio_durations(audio_durations)

    if not audio_durations.keys() == set(audio_ids):
        raise ValueError(
            f'audio_durations audio ids do not match audio ids in scores. '
            f'Missing ids: {set(audio_ids) - audio_durations.keys()}. '
            f'Additional ids: {audio_durations.keys() - set(audio_ids)}.'
        )
    dataset_duration = sum(audio_durations.values())

    intermediate_stats = intermediate_statistics(
        scores=scores, ground_truth=ground_truth,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )
    return psd_roc_from_intermediate_statistics(
        intermediate_stats,
        dataset_duration=dataset_duration,
        alpha_ct=alpha_ct, alpha_st=alpha_st,
        unit_of_time=unit_of_time, max_efpr=max_efpr
    )


def psd_roc_from_intermediate_statistics(
        scores_intermediate_statistics, dataset_duration,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
):
    """Computes Polyphonic Sound Detection ROC (PSD ROC) [1] from precomputed
    intermediate statistics curves.

    [1] C. Bilen, G. Ferroni, F. Tuveri, J. Azcarreta and S. Krstulovic,
    "A Framework for the Robust Evaluation of Sound Event Detection",
    in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
    2020, pp. 61–65

    Args:
        scores_intermediate_statistics (dict of tuples): tuple of scores array
            and dict of intermediate_statistics with the following key value
            pairs for each event class:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'cts' (2d np.ndarray): cross triggers with each of the other
                classes (second dim) for each score (first dim)
             'n_ref' (int): number of ground truth events
             't_ref_ofther' (list of float): total ground truth event durations
                for each of the other classes.
        dataset_duration (float): total dataset duration, i.e., the sum of the
            individual file durations.
        alpha_ct (float): parameter for penalizing cross triggers.
            More specifically, it is the weight of the cross trigger rate
            (averaged over all other classes) that is added to the False
            Positive Rate (FPR) yielding the effective FPR (eFPR).
        alpha_st (float): parameter for penalizing instability across classes.
            More specifically, it is the weight of the standard deviation of
            the per-class ROCs, that is subtracted from the mean of the
            per-class ROCs.
        unit_of_time (str): the unit of time \in {second, minute, hour} to be
            used for computation of the eFPR (which is defined as rate per unit
            of time). Default is hour.
        max_efpr (float): the maximum eFPR for which the system is evaluated,
            i.e., until where the Area under PSD ROC Curve is computed.

    Returns:
        etpr (1d np.ndarray): effective True Positive Rates.
        efpr (1d np.ndarray): effective False Positive Rates.
        single_class_psd_rocs (dict of tuples of 1d np.ndarrays):
            tuple of True Positive Rates and effective False Positive Rates
            for each event class.

    """

    if not isinstance(scores_intermediate_statistics, dict):
        raise ValueError(
            f'scores_intermediate_statistics must be dict '
            f'but {type(scores_intermediate_statistics)} was given.'
        )

    single_class_psd_rocs = _single_class_roc_from_intermediate_statistics(
        scores_intermediate_statistics,
        dataset_duration=dataset_duration,
        alpha_ct=alpha_ct, unit_of_time=unit_of_time, max_efpr=max_efpr,
    )
    effective_tp_rate, overall_effective_fp_rates = _psd_roc_from_single_class_rocs(single_class_psd_rocs, alpha_st=alpha_st, max_efpr=max_efpr)
    return effective_tp_rate, overall_effective_fp_rates, single_class_psd_rocs


def _single_class_roc_from_intermediate_statistics(
        scores_intermediate_statistics, dataset_duration,
        alpha_ct=.0, unit_of_time='hour', max_efpr=100.,
):
    """

    Args:
        scores_intermediate_statistics ((dict of) tuple): tuple of scores array and
            dict of intermediate_statistics with the following key value pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'cts' (2d np.ndarray): cross triggers with each of the other
                classes (second dim) for each score (first dim)
             'n_ref' (int): number of ground truth events
             't_ref_ofther' (list of float): total ground truth event durations
                for each of the other classes.
            If dict input is provided, keys are expected to be class names with
            corresponding scores/intermediate_statistics as values.
        dataset_duration (float): total dataset duration, i.e., the sum of the
            individual file durations.
        alpha_ct (float): parameter for penalizing cross triggers.
            More specifically, it is the weight of the cross trigger rate
            (averaged over all other classes) that is added to the False
            Positive Rate (FPR) yielding the effective FPR (eFPR).
        unit_of_time (str): the unit of time \in {second, minute, hour} to be
            used for computation of the eFPR (which is defined as rate per unit
            of time). Default is hour.
        max_efpr (float): the maximum eFPR for which the system is evaluated,
            i.e., until where the Area under PSD ROC Curve is computed.

    Returns:
        tp_ratio (1d np.ndarray): True Positive Ratios
        effective_fp_rate (1d np.ndarray): effective False Positive Rates
        scores (1d np.ndarray): corresponding scores that the decision
            threshold has to fall below.

    """
    if unit_of_time not in seconds_per_unit_of_time:
        raise ValueError(
            f'Invalid unit_of_time {unit_of_time}. Valid units are '
            f'{", ".join(list(seconds_per_unit_of_time.keys()))}.'
        )

    if isinstance(scores_intermediate_statistics, dict):
        return {
            cls: _single_class_roc_from_intermediate_statistics(
                scores_stats,
                dataset_duration=dataset_duration,
                alpha_ct=alpha_ct,
                unit_of_time=unit_of_time,
                max_efpr=max_efpr,
            ) for cls, scores_stats in (
                scores_intermediate_statistics.items())
        }

    scores, stats = scores_intermediate_statistics

    tp_ratio = stats['tps'] / max(stats['n_ref'], 1)
    fp_rate = stats['fps'] / dataset_duration
    if alpha_ct == .0:
        effective_fp_rate = fp_rate
    else:
        assert stats['cts'].shape == (len(scores), len(stats['t_ref_other'])), (
            stats['cts'].shape, len(scores), len(stats['t_ref_other']))
        ct_rates = [
            cts_i / max(t_ref_i, 1e-12)
            for cts_i, t_ref_i in zip(stats['cts'].T, stats['t_ref_other'])
        ]
        effective_fp_rate = fp_rate + alpha_ct * np.mean(ct_rates, axis=0)
    effective_fp_rate = effective_fp_rate * seconds_per_unit_of_time[unit_of_time]
    return _unique_cummax_sort(
        tp_ratio, effective_fp_rate, scores, max_efpr=max_efpr
    )


def _psd_roc_from_single_class_rocs(single_class_psd_rocs, alpha_st, max_efpr):
    tp_ratios, efp_rates, *_ = list(zip(*single_class_psd_rocs.values()))
    overall_effective_fp_rates = np.unique(np.sort(np.concatenate(efp_rates)))
    interpolated_tp_ratios = []
    for tpr, efpr in zip(tp_ratios, efp_rates):
        if len(tpr) == 1:
            # interp1d expects at least length of 2, which, however, isn't
            # necessary with bounds_error=False and fill_values. Therefore,
            # simply repeat arrays if length == 1
            tpr = tpr.repeat(2)
            efpr = efpr.repeat(2)
        interpolated_tp_ratios.append(
            interp1d(
                efpr, tpr, kind='previous',
                bounds_error=False, fill_value=(0, tpr[-1])
            )(overall_effective_fp_rates)
        )
    mu_tp = np.mean(interpolated_tp_ratios, axis=0)
    sigma_tp = np.std(interpolated_tp_ratios, axis=0)
    effective_tp_rate = mu_tp - alpha_st * sigma_tp
    effective_tp_rate = np.maximum(effective_tp_rate, 0.)
    if max_efpr is not None:
        effective_tp_rate = np.concatenate((
            effective_tp_rate, [effective_tp_rate[-1]]
        ))
        overall_effective_fp_rates = np.concatenate((
            overall_effective_fp_rates, [max_efpr]
        ))
    return effective_tp_rate, overall_effective_fp_rates


def _unique_cummax_sort(tp_ratio, effective_fp_rate, *other, max_efpr=None):
    sort_idx = sorted(
        np.arange(len(effective_fp_rate)).tolist(),
        key=lambda i: (effective_fp_rate[i], tp_ratio[i])
    )
    tp_ratio, effective_fp_rate, *other = [values[sort_idx] for values in [tp_ratio, effective_fp_rate, *other]]
    cummax_indices = cummax(tp_ratio)[1]
    tp_ratio, effective_fp_rate, *other = [values[cummax_indices] for values in [tp_ratio, effective_fp_rate, *other]]
    effective_fp_rate, unique_efpr_indices = np.unique(
        effective_fp_rate[::-1], return_index=True)
    unique_efpr_indices = - 1 - unique_efpr_indices
    tp_ratio, *other = [values[unique_efpr_indices] for values in [tp_ratio, *other]]
    if max_efpr is not None:
        cutoff_idx = get_first_index_where(effective_fp_rate, "gt", max_efpr)
        tp_ratio, effective_fp_rate, *other = [values[:cutoff_idx] for values in [tp_ratio, effective_fp_rate, *other]]
    return tp_ratio, effective_fp_rate, *other
