from sed_scores_eval.base_modules.error_rate import (
    single_error_rate_from_intermediate_statistics,
    best_error_rate_from_intermediate_statistics,
    error_rate_curve_from_intermediate_statistics
)
from sed_scores_eval.intersection_based.intermediate_statistics import (
    accumulated_intermediate_statistics,
)


def error_rate_curve(
        scores, ground_truth, *, deltas=None,
        dtc_threshold, gtc_threshold,
        time_decimals=6, num_jobs=1,
):
    """Compute intersection-based error rates with corresponding insertion rate,
    deletion rate and intermediate statistics for various operating points

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        deltas (dict of dicts of tuples): Must be deltas as returned by
            `accumulated_intermediate_statistics_from_deltas`. If not provided
            deltas are computed within this function. Providing deltas is useful
            if deltas are used repeatedly as, e.g., with bootstrapped evaluation,
            to save computing time.
        dtc_threshold (float): detection tolerance criterion threshold
        gtc_threshold (float): ground truth intersection criterion threshold
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, detections with an onset or offset right on the
            boundary of the collar may be falsely counted as false detection
            because of small deviations due to limited floating point precision.
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns: (all arrays sorted by corresponding score)
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
    intermediate_stats, _ = accumulated_intermediate_statistics(
        scores=scores, ground_truth=ground_truth, deltas=deltas,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )
    return error_rate_curve_from_intermediate_statistics(intermediate_stats)


def error_rate(
        scores, ground_truth, threshold, *, deltas=None,
        dtc_threshold, gtc_threshold,
        time_decimals=6, num_jobs=1,
):
    """Compute intersection-based error rates with corresponding insertion
    rate, deletion rate and intermediate statistics intermediate statistics for
    a specific decision threshold

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        threshold ((dict of) float): threshold that is to be evaluated.
        deltas (dict of dicts of tuples): Must be deltas as returned by
            `accumulated_intermediate_statistics_from_deltas`. If not provided
            deltas are computed within this function. Providing deltas is useful
            if deltas are used repeatedly as, e.g., with bootstrapped evaluation,
            to save computing time.
        dtc_threshold (float): detection tolerance criterion threshold
        gtc_threshold (float): ground truth intersection criterion threshold
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, detections with an onset or offset right on the
            boundary of the collar may be falsely counted as false detection
            because of small deviations due to limited floating point precision.
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

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
    intermediate_stats, _ = accumulated_intermediate_statistics(
        scores=scores, ground_truth=ground_truth, deltas=deltas,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )
    return single_error_rate_from_intermediate_statistics(
        intermediate_stats, threshold=threshold,
    )


def best_error_rate(
        scores, ground_truth, *, deltas=None,
        dtc_threshold, gtc_threshold,
        max_insertion_rate=None, max_deletion_rate=None,
        time_decimals=6, num_jobs=1,
):
    """Get the best possible (macro-averaged) intersection-based error rate
    with corresponding insertion rate, deletion rate, intermediate statistics
    and decision threshold

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        deltas (dict of dicts of tuples): Must be deltas as returned by
            `accumulated_intermediate_statistics_from_deltas`. If not provided
            deltas are computed within this function. Providing deltas is useful
            if deltas are used repeatedly as, e.g., with bootstrapped evaluation,
            to save computing time.
        dtc_threshold (float): detection tolerance criterion threshold
        gtc_threshold (float): ground truth intersection criterion threshold
        max_insertion_rate: a maximal insertion rate that must not be exceeded.
        max_deletion_rate: a maximal deletion rate that must not be exceeded.
            If the constraint(s) cannot be achieved at any threshold, however,
            er, ir, dr and threshold of 1,0,1,inf are returned.
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, detections with an onset or offset right on the
            boundary of the collar may be falsely counted as false detection
            because of small deviations due to limited floating point precision.
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

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
    intermediate_stats, _ = accumulated_intermediate_statistics(
        scores=scores, ground_truth=ground_truth, deltas=deltas,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )
    return best_error_rate_from_intermediate_statistics(
        intermediate_stats, max_insertion_rate=max_insertion_rate,
        max_deletion_rate=max_deletion_rate,
    )
