import numpy as np
from pathlib import Path
from sed_scores_eval.utils.array_ops import cummax, get_first_index_where
from sed_scores_eval.base_modules.io import parse_inputs, read_audio_durations
from sed_scores_eval.utils.auc import staircase_auc
from sed_scores_eval.intersection_based.intermediate_statistics import intermediate_statistics
from scipy.interpolate import interp1d

seconds_per_unit_of_time = {
    'second': 1.,
    'minute': 60.,
    'hour': 3600.,
}


def psds(
        scores, ground_truth, audio_durations, dtc_threshold, gtc_threshold,
        cttc_threshold=None, alpha_ct=.0, alpha_st=.0,
        unit_of_time='hour', max_efpr=100.,
        time_decimals=6,
):
    """

    Args:
        scores:
        ground_truth:
        audio_durations:
        dtc_threshold:
        gtc_threshold:
        cttc_threshold:
        alpha_ct:
        alpha_st:
        unit_of_time:
        max_efpr:
        time_decimals:

    Returns:

    """
    effective_tp_rate, effective_fp_rate, single_class_psds_rocs = psd_roc(
        scores=scores, ground_truth=ground_truth, audio_durations=audio_durations,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold, alpha_ct=alpha_ct, alpha_st=alpha_st,
        unit_of_time=unit_of_time, max_efpr=max_efpr,
        time_decimals=time_decimals,
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
        time_decimals=6,
):

    if alpha_ct == 0.:
        assert cttc_threshold is None, cttc_threshold
    else:
        assert cttc_threshold is not None

    scores, ground_truth, keys = parse_inputs(scores, ground_truth)
    assert isinstance(audio_durations, (dict, str, Path)), type(audio_durations)
    if isinstance(audio_durations, (str, Path)):
        audio_durations = Path(audio_durations)
        assert audio_durations.is_file(), audio_durations
        audio_durations = read_audio_durations(audio_durations)
    assert audio_durations.keys() == set(keys), (
        set(keys) - audio_durations.keys(), audio_durations.keys() - set(keys))
    dataset_duration = sum(audio_durations.values())

    intermediate_stats = intermediate_statistics(
        scores=scores, ground_truth=ground_truth,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
        time_decimals=time_decimals,
    )
    return psd_roc_from_intermediate_statistics(
        intermediate_stats,
        dataset_duration=dataset_duration,
        alpha_ct=alpha_ct, alpha_st=alpha_st,
        unit_of_time=unit_of_time, max_efpr=max_efpr
    )


def psd_roc_from_intermediate_statistics(
        intermediate_stats, dataset_duration,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
):
    """

    Args:
        intermediate_stats:
        dataset_duration:
        alpha_ct:
        alpha_st:
        unit_of_time:
        max_efpr:

    Returns:

    """

    single_class_psd_rocs = _single_class_roc_from_intermediate_statistics(
        intermediate_stats,
        dataset_duration=dataset_duration,
        alpha_ct=alpha_ct, unit_of_time=unit_of_time, max_efpr=max_efpr,
    )

    tp_ratios, efp_rates, _ = list(zip(*single_class_psd_rocs.values()))
    overall_effective_fp_rates = np.unique(np.sort(np.concatenate(efp_rates)))
    interpolated_tp_ratios = [
        interp1d(
            efpr, tpr, kind='previous',
            bounds_error=False, fill_value=(0, tpr[-1])
        )(overall_effective_fp_rates)
        for tpr, efpr in zip(tp_ratios, efp_rates)
    ]
    mu_tp = np.mean(interpolated_tp_ratios, axis=0)
    sigma_tp = np.std(interpolated_tp_ratios, axis=0)
    effective_tp_rate = mu_tp - alpha_st * sigma_tp
    effective_tp_rate = np.maximum(effective_tp_rate, 0.)
    if max_efpr is not None:
        effective_tp_rate = np.array(
            effective_tp_rate.tolist() + [effective_tp_rate[-1]])
        overall_effective_fp_rates = np.array(
            overall_effective_fp_rates.tolist() + [max_efpr]
        )
    return effective_tp_rate, overall_effective_fp_rates, single_class_psd_rocs


def _single_class_roc_from_intermediate_statistics(
        intermediate_stats, dataset_duration,
        alpha_ct=.0, unit_of_time='hour', max_efpr=100.,
):
    """

    Args:
        intermediate_stats:
        dataset_duration:
        alpha_ct:
        unit_of_time:
        max_efpr:

    Returns:

    """
    if unit_of_time not in seconds_per_unit_of_time:
        raise ValueError(
            f'Invalid unit_of_time {unit_of_time}. Valid units are '
            f'{", ".join(list(seconds_per_unit_of_time.keys()))}.'
        )

    if isinstance(intermediate_stats, dict):
        return {
            cls: _single_class_roc_from_intermediate_statistics(
                scores_stats,
                dataset_duration=dataset_duration,
                alpha_ct=alpha_ct,
                unit_of_time=unit_of_time,
                max_efpr=max_efpr,
            ) for cls, scores_stats in (
                intermediate_stats.items())
        }

    scores, stats = intermediate_stats

    tp_ratio = stats['tps'] / stats['n_ref']
    fp_rate = stats['fps'] / dataset_duration
    if alpha_ct == .0:
        effective_fp_rate = fp_rate
    else:
        assert stats['cts'].shape == (len(scores), len(stats['t_ref_other'])), (
            stats['cts'].shape, len(scores), len(stats['t_ref_other']))
        ct_rates = [
            cts_i / t_ref_i
            for cts_i, t_ref_i in zip(stats['cts'].T, stats['t_ref_other'])
        ]
        effective_fp_rate = fp_rate + alpha_ct * np.mean(ct_rates, axis=0) * len(ct_rates)/(len(ct_rates)+1)
    sort_idx = sorted(
        np.arange(len(effective_fp_rate)).tolist(),
        key=lambda i: (effective_fp_rate[i], tp_ratio[i])
    )
    effective_fp_rate = effective_fp_rate[sort_idx]
    tp_ratio = tp_ratio[sort_idx]
    scores = scores[sort_idx]
    _, cummax_indices = cummax(tp_ratio)
    cummax_indices = np.unique(cummax_indices)
    tp_ratio = tp_ratio[cummax_indices]
    effective_fp_rate = (
        effective_fp_rate[cummax_indices]
        * seconds_per_unit_of_time[unit_of_time]
    )
    scores = scores[cummax_indices]
    effective_fp_rate, unique_efpr_indices = np.unique(
        effective_fp_rate[::-1], return_index=True)
    unique_efpr_indices = - 1 - unique_efpr_indices
    tp_ratio = tp_ratio[unique_efpr_indices]
    scores = scores[unique_efpr_indices]
    if max_efpr is not None:
        cutoff_idx = get_first_index_where(effective_fp_rate, "gt", max_efpr)
        tp_ratio = tp_ratio[:cutoff_idx]
        effective_fp_rate = effective_fp_rate[:cutoff_idx]
        scores = scores[:cutoff_idx]
    return tp_ratio, effective_fp_rate, scores
