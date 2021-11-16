
from sed_scores_eval.collar_based.intermediate_statistics import intermediate_statistics
import sed_scores_eval.base_modules.fscore as base


def fscore(
        scores, ground_truth, threshold, *,
        onset_collar, offset_collar, offset_collar_rate=0., beta=1.,
        time_decimals=6,
):
    intermediate_stats = intermediate_statistics(
        scores=scores, ground_truth=ground_truth,
        onset_collar=onset_collar, offset_collar=offset_collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals,
    )
    return base.fscore_from_intermediate_statistics(
        intermediate_stats, threshold=threshold, beta=beta,
    )


def best_fscore(
        scores, ground_truth, *,
        onset_collar, offset_collar, offset_collar_rate=0.,
        min_precision=0., min_recall=0., beta=1.,
        time_decimals=6,
):
    intermediate_stats = intermediate_statistics(
        scores=scores, ground_truth=ground_truth,
        onset_collar=onset_collar, offset_collar=offset_collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals,
    )
    return base.best_fscore_from_intermediate_statistics(
        intermediate_stats,
        min_precision=min_precision, min_recall=min_recall, beta=beta,
    )


def precision_recall_curves(
        scores, ground_truth, *,
        onset_collar, offset_collar, offset_collar_rate=0.,
        time_decimals=6,
):
    intermediate_stats = intermediate_statistics(
        scores=scores, ground_truth=ground_truth,
        onset_collar=onset_collar, offset_collar=offset_collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals,
    )
    return base.precision_recall_curve_from_intermediate_statistics(
        intermediate_stats
    )
