from sed_scores_eval.collar_based.intermediate_statistics import intermediate_statistics
from sed_scores_eval.base_modules.fscore import (
    single_fscore_from_intermediate_statistics,
    best_fscore_from_intermediate_statistics,
    precision_recall_curve_from_intermediate_statistics,
    fscore_curve_from_intermediate_statistics
)


def fscore(
        scores, ground_truth, threshold, *,
        onset_collar, offset_collar, offset_collar_rate=0., beta=1.,
        time_decimals=6,
):
    """Get collar-based f-score with corresponding precision, recall and
    intermediate statistics for a specific decision threshold

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        threshold ((dict of) float): threshold that is to be evaluated.
        onset_collar (float): allowed onset deviation in seconds
        offset_collar (float): (at least) allowed offset deviation in seconds
        offset_collar_rate (float): (at least) allowed offset deviation as a
            ratio of the length of the ground truth event, with the actual
            allowed offset deviation being:
            offset_collar_for_gt_event = max(
                offset_collar, offset_collar_rate*length_of_gt_event_in_seconds
            )
        beta: \beta parameter for f-score computation
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, detections with an onset or offset right on the
            boundary of the collar may be falsely counted as false detection
            because of small deviations due to limited floating point precision.

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
    intermediate_stats = intermediate_statistics(
        scores=scores, ground_truth=ground_truth,
        onset_collar=onset_collar, offset_collar=offset_collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals,
    )
    return single_fscore_from_intermediate_statistics(
        intermediate_stats, threshold=threshold, beta=beta,
    )


def best_fscore(
        scores, ground_truth, *,
        onset_collar, offset_collar, offset_collar_rate=0.,
        min_precision=0., min_recall=0., beta=1.,
        time_decimals=6,
):
    """Get the best possible (macro-averaged) f-score with corresponding
    precision, recall, intermediate statistics and decision threshold

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        onset_collar (float): allowed onset deviation in seconds
        offset_collar (float): (at least) allowed offset deviation in seconds
        offset_collar_rate (float): (at least) allowed offset deviation as a
            ratio of the length of the ground truth event, with the actual
            allowed offset deviation being:
            offset_collar_for_gt_event = max(
                offset_collar, offset_collar_rate*length_of_gt_event_in_seconds
            )
        min_precision: the minimum precision that should be achieved. If the
            provided precision cannot be achieved at any threshold an fscore,
            precision and recall of 0 and threshold of np.inf is returned.
        min_recall: the minimum recall that should be achieved. If the
            provided recall cannot be achieved at any threshold an fscore,
            precision and recall of 0 and threshold of np.inf is returned.
        beta: \beta parameter for f-score computation
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, detections with an onset or offset right on the
            boundary of the collar may be falsely counted as false detection
            because of small deviations due to limited floating point precision.

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
    intermediate_stats = intermediate_statistics(
        scores=scores, ground_truth=ground_truth,
        onset_collar=onset_collar, offset_collar=offset_collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals,
    )
    return best_fscore_from_intermediate_statistics(
        intermediate_stats, beta=beta,
        min_precision=min_precision, min_recall=min_recall,
    )


def precision_recall_curve(
        scores, ground_truth, *,
        onset_collar, offset_collar, offset_collar_rate=0.,
        time_decimals=6,
):
    """Compute collar-based precision-recall curve [1].

    [1] J.Ebbers, R.Serizel, and R.Haeb-Umbach
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
        onset_collar (float): allowed onset deviation in seconds
        offset_collar (float): (at least) allowed offset deviation in seconds
        offset_collar_rate (float): (at least) allowed offset deviation as a
            ratio of the length of the ground truth event, with the actual
            allowed offset deviation being:
            offset_collar_for_gt_event = max(
                offset_collar, offset_collar_rate*length_of_gt_event_in_seconds
            )
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, detections with an onset or offset right on the
            boundary of the collar may be falsely counted as false detection
            because of small deviations due to limited floating point precision.

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
    intermediate_stats = intermediate_statistics(
        scores=scores, ground_truth=ground_truth,
        onset_collar=onset_collar, offset_collar=offset_collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals,
    )
    return precision_recall_curve_from_intermediate_statistics(
        intermediate_stats
    )


def fscore_curve(
        scores, ground_truth, *,
        onset_collar, offset_collar, offset_collar_rate=0.,
        beta=1., time_decimals=6,
):
    """Compute collar-based f-scores with corresponding precisions, recalls and
    intermediate statistics for various operating points

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        onset_collar (float): allowed onset deviation in seconds
        offset_collar (float): (at least) allowed offset deviation in seconds
        offset_collar_rate (float): (at least) allowed offset deviation as a
            ratio of the length of the ground truth event, with the actual
            allowed offset deviation being:
            offset_collar_for_gt_event = max(
                offset_collar, offset_collar_rate*length_of_gt_event_in_seconds
            )
        beta: \beta parameter for f-score computation
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high, detections with an onset or offset right on the
            boundary of the collar may be falsely counted as false detection
            because of small deviations due to limited floating point precision.

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
    intermediate_stats = intermediate_statistics(
        scores=scores, ground_truth=ground_truth,
        onset_collar=onset_collar, offset_collar=offset_collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals,
    )
    return fscore_curve_from_intermediate_statistics(
        intermediate_stats, beta=beta,
    )
