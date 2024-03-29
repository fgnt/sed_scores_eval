import numpy as np
from sed_scores_eval.base_modules.curves import get_curve_idx_for_threshold, xsort


def error_rate_curve_from_intermediate_statistics(
        scores_intermediate_statistics
):
    """Compute error rate curve from intermediate_statistics

    Args:
        scores_intermediate_statistics ((dict of) tuple): tuple of scores array
            and dict of intermediate_statistics with the following key value
            pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'n_ref' (int): number of ground truth events
            If dict input is provided, keys are expected to be class names with
            corresponding scores/intermediate_statistics as values.

    Returns: (all arrays sorted by corresponding recall)
        error_rates ((dict of) 1d np.ndarray): error rates for all operating points
        insertion_rate ((dict of) 1d np.ndarray): insertion rates for all operating points
        deletion_rates ((dict of) 1d np.ndarray): deletion rates for all operating points
        scores ((dict of) 1d np.ndarray): score values that the threshold has to
            fall below to obtain corresponding error rate.
        intermediate_statistics ((dict of) dict): dict of
            intermediate_statistics with the following key value pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'n_ref' (int): number of ground truth events

    """
    if isinstance(scores_intermediate_statistics, dict):
        error_rate_curves = {}
        for class_name, scores_stats in scores_intermediate_statistics.items():
            error_rate_curves[class_name] = error_rate_curve_from_intermediate_statistics(scores_stats)
        return error_rate_curves

    if not isinstance(scores_intermediate_statistics, (list, tuple)):
        raise ValueError(
            f'scores_intermediate_statistics must be list/tuple of length 2, '
            f'but {type(scores_intermediate_statistics)} was given.'
        )
    if len(scores_intermediate_statistics) != 2:
        raise ValueError(
            f'scores_intermediate_statistics must be list/tuple of length 2, '
            f'but list/tuple of length {len(scores_intermediate_statistics)}'
            f'was given.'
        )
    scores, stats = scores_intermediate_statistics
    if not isinstance(stats, dict):
        raise ValueError(
            f'intermediate_statistics must be dict but {type(stats)} was given.')
    if not all([key in stats for key in ['tps', 'fps', 'n_ref']]):
        raise ValueError(
            f'intermediate_statistics must contain keys "tps", "fps" and "n_ref". '
            f'Provided keys: {sorted(stats.keys())}.'
        )
    insertions = stats['fps']
    deletions = stats['n_ref'] - stats['tps']
    errors = insertions + deletions
    er = errors / stats['n_ref']
    ir = insertions / stats['n_ref']
    dr = deletions / stats['n_ref']
    er, scores, ir, dr, stats = xsort(er, scores, ir, dr, stats)
    return er, ir, dr, scores, stats


def best_error_rate_from_intermediate_statistics(
        scores_intermediate_statistics,
        max_insertion_rate=None, max_deletion_rate=None,
):
    """Get the best possible (macro-average) error rate with corresponding
    insertion rate, deletion_rate, intermediate statistics and decision
    threshold

    Args:
        scores_intermediate_statistics ((dict of) tuple): tuple of scores array
            and dict of intermediate_statistics with the following key value
            pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'n_ref' (int): number of ground truth events
            If dict input is provided keys are expected to be class names with
            corresponding scores/intermediate_statistics as values.
        max_insertion_rate: a maximal insertion rate that must not be exceeded.
        max_deletion_rate: a maximal deletion rate that must not be exceeded.
            If the constraint(s) cannot be achieved at any threshold, however,
            er, ir, dr and threshold of 1,0,1,inf are returned.

    Returns:
        error_rate ((dict of) float): best achievable error_rate value
        insertion_rate ((dict of) float): insertion rate at best error rate
        deletion_rate ((dict of) float): deletion rate at best error rate
        threshold ((dict of) float): threshold to obtain best error rate which
            is centered between the score that the threshold has to fall below
            and the next smaller score which results in different intermediate
            statistics.
        intermediate_statistics ((dict of) dict): dict of
            intermediate_statistics with the following key value pairs:
            'tps' (int): true positive count at best error rate
            'fps' (int): false positive count at best error rate
            'n_ref' (int): number of ground truth events

    """
    if isinstance(scores_intermediate_statistics, dict):
        er, ir, dr, thresholds, intermediate_stats = {}, {}, {}, {}, {}
        for class_name, scores_stats in scores_intermediate_statistics.items():
            (
                er[class_name], ir[class_name], dr[class_name],
                thresholds[class_name], intermediate_stats[class_name]
            ) = best_error_rate_from_intermediate_statistics(
                scores_stats,
                max_insertion_rate=max_insertion_rate,
                max_deletion_rate=max_deletion_rate,
            )
        er['macro_average'] = np.mean([er[class_name] for class_name in er])
        ir['macro_average'] = np.mean([ir[class_name] for class_name in ir])
        dr['macro_average'] = np.mean([dr[class_name] for class_name in dr])
        return er, ir, dr, thresholds, intermediate_stats
    er, ir, dr, scores, intermediate_stats = \
        error_rate_curve_from_intermediate_statistics(
            scores_intermediate_statistics,
        )
    if max_insertion_rate is not None:
        er[ir < max_insertion_rate] = np.inf
    if max_deletion_rate is not None:
        er[dr < max_insertion_rate] = np.inf
    best_idx = len(er) - 1 - np.argmin(er[::-1], axis=0)
    threshold = (
        (scores[best_idx] + scores[best_idx-1])/2 if best_idx > 0 else -np.inf
    )

    def _recursive_get_item(stats, idx):
        if isinstance(stats, dict):
            return {key: _recursive_get_item(stats[key], idx) for key in stats}
        if np.isscalar(stats):
            return stats
        return stats[idx]
    return (
        er[best_idx], ir[best_idx], dr[best_idx], threshold,
       _recursive_get_item(intermediate_stats, best_idx)
    )


def single_error_rate_from_intermediate_statistics(
        scores_intermediate_statistics, threshold):
    """Get error rate with corresponding insertion rate, deletion rate and
    intermediate statistics for a specific decision threshold

    Args:
        scores_intermediate_statistics ((dict of) tuple): tuple of scores array
            and dict of intermediate_statistics with the following key value
            pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'n_ref' (int): number of ground truth events
            If dict input is provided keys are expected to be class names with
            corresponding scores/intermediate_statistics as values.
        threshold ((dict of) float): threshold that is to be evaluated.

    Returns:
        error_rate ((dict of) float): error rate for threshold
        insertion_rate ((dict of) float): insertion rate for threshold
        deletion rate ((dict of) float): deletion rate for threshold
        intermediate_statistics ((dict of) dict): dict of
            intermediate_statistics with the following key value pairs:
            'tps' (int): true positive count for threshold
            'fps' (int): false positive count for threshold
            'n_ref' (int): number of ground truth events

    """
    if isinstance(threshold, dict):
        if not all([np.isscalar(thr) for thr in threshold]):
            raise ValueError('All values of thresholds dict must be scalars')
    elif not np.isscalar(threshold):
        raise ValueError(
            f'threshold must be (dict of) scalar(s) but '
            f'{type(threshold)} was given.'
        )
    if isinstance(scores_intermediate_statistics, dict):
        er, ir, dr, intermediate_stats = {}, {}, {}, {}
        for class_name, scores_stats in scores_intermediate_statistics.items():
            (
                er[class_name], ir[class_name], dr[class_name],
                intermediate_stats[class_name]
            ) = single_error_rate_from_intermediate_statistics(
                scores_stats,
                threshold[class_name]
                if isinstance(threshold, dict)
                else threshold,
            )
        er['macro_average'] = np.mean([er[class_name] for class_name in er])
        ir['macro_average'] = np.mean([ir[class_name] for class_name in ir])
        dr['macro_average'] = np.mean([dr[class_name] for class_name in dr])
        return er, ir, dr, intermediate_stats
    error_rate_curve = error_rate_curve_from_intermediate_statistics(
        scores_intermediate_statistics
    )
    er, ir, dr, scores, intermediate_stats = error_rate_curve
    return _single_er_from_er_curve(
        er, ir, dr, scores, threshold,
        intermediate_statistics=intermediate_stats
    )


def _single_er_from_er_curve(
        error_rates, insertion_rate, deletion_rate, scores, threshold,
        intermediate_statistics=None
):
    idx = get_curve_idx_for_threshold(scores, threshold)
    error_rate = error_rates[idx]
    insertion_rate = insertion_rate[idx]
    deletion_rate = deletion_rate[idx]
    if intermediate_statistics is None:
        return error_rate, insertion_rate, deletion_rate
    else:
        return error_rate, insertion_rate, deletion_rate, {
            key: stat if np.isscalar(stat) else stat[idx]
            for key, stat in intermediate_statistics.items()
        }


def error_rate_from_sed_eval_metrics(sed_eval_metrics):
    """extract class-wise and macro-averaged error,insertion and deletion rates
    from sed_eval metrics object

    Args:
        sed_eval_metrics:

    Returns:
        er (dict of float): error rates
        ir (dict of float): insertion rates
        dr (dict of float): deletion rates

    """
    er = {}
    ir = {}
    dr = {}
    sed_eval_results_classwise = sed_eval_metrics.results_class_wise_metrics()
    for key in sed_eval_results_classwise:
        er[key] = sed_eval_results_classwise[key]['error_rate']['error_rate']
        ir[key] = sed_eval_results_classwise[key]['error_rate']['insertion_rate']
        dr[key] = sed_eval_results_classwise[key]['error_rate']['deletion_rate']
    sed_eval_results_macro = sed_eval_metrics.results_class_wise_average_metrics()
    er['macro_average'] = sed_eval_results_macro['error_rate']['error_rate']
    ir['macro_average'] = sed_eval_results_macro['error_rate']['insertion_rate']
    dr['macro_average'] = sed_eval_results_macro['error_rate']['deletion_rate']
    sed_eval_results_micro = sed_eval_metrics.results_overall_metrics()
    er['micro_average'] = sed_eval_results_micro['error_rate']['error_rate']
    ir['micro_average'] = sed_eval_results_micro['error_rate']['insertion_rate']
    dr['micro_average'] = sed_eval_results_micro['error_rate']['deletion_rate']
    return er, ir, dr
