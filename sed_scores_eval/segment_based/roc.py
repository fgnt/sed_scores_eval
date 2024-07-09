from sed_scores_eval.base_modules.roc import (
    roc_curve_from_intermediate_statistics,
    auroc_from_intermediate_statistics
)
from sed_scores_eval.base_modules.io import parse_inputs
from sed_scores_eval.base_modules.bootstrap import bootstrap
from sed_scores_eval.segment_based.intermediate_statistics import (
    accumulated_intermediate_statistics, intermediate_statistics_deltas)


def roc_curve(
        scores, ground_truth, audio_durations, *, deltas=None,
        segment_length, time_decimals=6, num_jobs=1,
):
    """

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        audio_durations: The duration of each audio file in the evaluation set.
        deltas (dict of dicts of tuples): Must be deltas as returned by
            `accumulated_intermediate_statistics_from_deltas`. If not provided,
            deltas are computed within this function. Providing deltas is useful
            if deltas are used repeatedly as, e.g., with bootstrapped evaluation,
            to save computing time.
        segment_length: the segment length of the segments that are to be
            evaluated.
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high detected or ground truth events that have
            onsets or offsets right on a segment boundary may swap over to the
            adjacent segment because of small deviations due to limited
            floating point precision.
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
        scores=scores, ground_truth=ground_truth,
        audio_durations=audio_durations, deltas=deltas,
        segment_length=segment_length, time_decimals=time_decimals,
        num_jobs=num_jobs,
    )
    return roc_curve_from_intermediate_statistics(intermediate_stats)


def auroc(
        scores, ground_truth, audio_durations, *, deltas=None,
        segment_length, max_fpr=None, mcclish_correction=False,
        time_decimals=6, num_jobs=1,
):
    """compute area under ROC curve

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        audio_durations: The duration of each audio file in the evaluation set.
        deltas (dict of dicts of tuples): Must be deltas as returned by
            `accumulated_intermediate_statistics_from_deltas`. If not provided,
            deltas are computed within this function. Providing deltas is useful
            if deltas are used repeatedly as, e.g., with bootstrapped evaluation,
            to save computing time.
        segment_length: the segment length of the segments that are to be
            evaluated.
        max_fpr (float): maximum false positive rate up to which to compute partial auc
        mcclish_correction: whether to use mcclish correction to get result back
            into [0.5,1.] range when using max_fpr.
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high detected or ground truth events that have
            onsets or offsets right on a segment boundary may swap over to the
            adjacent segment because of small deviations due to limited
            floating point precision.
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns:
        auroc ((dict of) float): mean and class-wise area under ROC curve
        roc_curve: ROC (TPR-FPR) curve(s) as provided by
            `roc_curve_from_intermediate_statistics`

    """
    intermediate_stats, _ = accumulated_intermediate_statistics(
        scores=scores, ground_truth=ground_truth,
        audio_durations=audio_durations, deltas=deltas,
        segment_length=segment_length, time_decimals=time_decimals,
        num_jobs=num_jobs,
    )
    return auroc_from_intermediate_statistics(
        intermediate_stats, max_fpr=max_fpr, mcclish_correction=mcclish_correction
    )


def bootstrapped_auroc(
        scores, ground_truth, audio_durations, *, deltas=None,
        segment_length, max_fpr=None, time_decimals=6,
        n_bootstrap_samples=100, num_jobs=1,
):
    if scores is not None:
        scores, ground_truth, audio_ids = parse_inputs(scores, ground_truth)
    return bootstrap(
        auroc, scores=scores, deltas=deltas,
        deltas_fn=intermediate_statistics_deltas, num_jobs=num_jobs,
        deltas_fn_kwargs=dict(
            ground_truth=ground_truth,
            audio_durations=audio_durations,
            segment_length=segment_length,
            time_decimals=time_decimals,
        ),
        eval_fn_kwargs=dict(
            max_fpr=max_fpr,
        ),
        n_bootstrap_samples=n_bootstrap_samples,
    )
