from sed_scores_eval.base_modules.precision_recall import (
    precision_recall_curve_from_intermediate_statistics,
    fscore_curve_from_intermediate_statistics,
    single_fscore_from_intermediate_statistics,
    best_fscore_from_intermediate_statistics,
    average_precision_from_intermediate_statistics,
)
from sed_scores_eval.clip_based.intermediate_statistics import accumulated_intermediate_statistics


def precision_recall_curve(scores, ground_truth, *, deltas=None, num_jobs=1):
    """Compute clip-based precision-recall curve.

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
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns: (all arrays sorted by corresponding recall)
        precisions ((dict of) 1d np.ndarray): precision values for all operating points
        recalls ((dict of) 1d np.ndarray): recall values for all operating points
        scores ((dict of) 1d np.ndarray): score values that the threshold has to
            fall below to obtain corresponding precision-recall pairs
        intermediate_statistics ((dict of) dict): dict of
            intermediate_statistics with the following key value pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'n_ref' (int): number of ground truth events

    """
    intermediate_stats, _ = accumulated_intermediate_statistics(
        scores=scores, ground_truth=ground_truth, deltas=deltas,
        num_jobs=num_jobs,
    )
    return precision_recall_curve_from_intermediate_statistics(
        intermediate_stats
    )


def fscore_curve(
        scores, ground_truth, *, deltas=None, beta=1., num_jobs=1,
):
    """Compute clip-based f-scores with corresponding precisions, recalls and
    intermediate statistics for various operating points

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
        beta: \beta parameter for f-score computation
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns: (all arrays sorted by corresponding score)
        f_beta ((dict of) 1d np.ndarray): f-score values  for all operating
            points
        precisions ((dict of) 1d np.ndarray): precision values for all operating points
        recalls ((dict of) 1d np.ndarray): recall values for all operating points
        scores ((dict of) 1d np.ndarray): score values that the threshold has to
            fall below to obtain corresponding precision-recall pairs
        intermediate_statistics ((dict of) dict): dict of
            intermediate_statistics with the following key value pairs:
            'tps': 1d np.ndarray of true positive counts for each score
            'fps': 1d np.ndarray of false positive counts for each score
            'n_ref': integer number of ground truth events

    """
    intermediate_stats, _ = accumulated_intermediate_statistics(
        scores=scores, ground_truth=ground_truth, deltas=deltas,
        num_jobs=num_jobs,
    )
    return fscore_curve_from_intermediate_statistics(
        intermediate_stats, beta=beta,
    )


def fscore(
        scores, ground_truth, threshold, *, deltas=None, beta=1., num_jobs=1,
):
    """Get clip-based f-score with corresponding precision, recall and
    intermediate statistics for a specific decision threshold

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
        threshold ((dict of) float): threshold that is to be evaluated.
        beta: \beta parameter for f-score computation
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns:
        fscore ((dict of) float): fscore value for threshold
        precision ((dict of) float): precision value for threshold
        recall ((dict of) float): recall value for threshold
        intermediate_statistics ((dict of) dict): dict of
            intermediate_statistics with the following key value pairs:
            'tps' (int): true positive count for threshold
            'fps' (int): false positive count for threshold
            'n_ref' (int): number of ground truth events

    """
    intermediate_stats, _ = accumulated_intermediate_statistics(
        scores=scores, ground_truth=ground_truth, deltas=deltas,
        num_jobs=num_jobs,
    )
    return single_fscore_from_intermediate_statistics(
        intermediate_stats, threshold=threshold, beta=beta,
    )


def best_fscore(
        scores, ground_truth, *, deltas=None,
        min_precision=0., min_recall=0., beta=1., num_jobs=1
):
    """Get the best possible (macro-averaged) clip-based f-score with
    corresponding precision, recall, intermediate statistics and decision
    threshold

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
        min_precision: the minimum precision that must be achieved.
        min_recall: the minimum recall that must be achieved. If the
            constraint(s) cannot be achieved at any threshold, however,
            fscore, precision, recall and threshold of 0,1,0,inf are returned.
        beta: \beta parameter for f-score computation
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns:
        f_beta ((dict of) float): best achievable f-score value
        precision ((dict of) float): precision value at best fscore
        recall ((dict of) float): recall value at best fscore
        threshold ((dict of) float): threshold to obtain best fscore which is
            centered between the score that the threshold has to fall below
            and the next smaller score which results in different intermediate
            statistics.
        intermediate_statistics ((dict of) dict): dict of
            intermediate_statistics with the following key value pairs:
            'tps' (int): true positive count at best fscore
            'fps' (int): false positive count at best fscore
            'n_ref' (int): number of ground truth events

    """
    intermediate_stats, _ = accumulated_intermediate_statistics(
        scores=scores, ground_truth=ground_truth, deltas=deltas,
        num_jobs=num_jobs,
    )
    return best_fscore_from_intermediate_statistics(
        intermediate_stats, beta=beta,
        min_precision=min_precision, min_recall=min_recall,
    )


def average_precision(scores, ground_truth, *, deltas=None, num_jobs=1):
    """Compute average precision

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
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns:
        average_precision ((dict of) float): mean and class-wise average
            precisions
        pr_curve: Precision-Recall curve(s) as provided by
            `precision_recall_curve_from_intermediate_statistics`

    """
    intermediate_stats, _ = accumulated_intermediate_statistics(
        scores=scores, ground_truth=ground_truth, deltas=deltas,
        num_jobs=num_jobs,
    )
    return average_precision_from_intermediate_statistics(intermediate_stats)
