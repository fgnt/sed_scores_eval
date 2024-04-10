from sed_scores_eval.base_modules.roc import (
    roc_curve_from_intermediate_statistics,
    auroc_from_intermediate_statistics
)
from sed_scores_eval.clip_based.intermediate_statistics import accumulated_intermediate_statistics


def roc_curve(scores, ground_truth, *, deltas=None, num_jobs=1):
    """

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

    Returns: (all arrays sorted by corresponding false positive rate)
        true_positive_rates ((dict of) 1d np.ndarray): TPR values for all operating points
        false_positive_rates ((dict of) 1d np.ndarray): FPR values for all operating points
        scores ((dict of) 1d np.ndarray): score values that the threshold has to
            fall below to obtain corresponding TPR-FPR pairs
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
    return roc_curve_from_intermediate_statistics(intermediate_stats)


def auroc(scores, ground_truth, *, deltas=None, max_fpr=None, num_jobs=1):
    """compute area under ROC curve

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
        max_fpr (float): maximum false positive rate up to which to compute partial auc
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns:
        auroc ((dict of) float): mean and class-wise area under ROC curve
        roc_curve: ROC (TPR-FPR) curve(s) as provided by
            `roc_curve_from_intermediate_statistics`

    """
    intermediate_stats, _ = accumulated_intermediate_statistics(
        scores=scores, ground_truth=ground_truth, deltas=deltas,
        num_jobs=num_jobs,
    )
    return auroc_from_intermediate_statistics(intermediate_stats, max_fpr=max_fpr)
