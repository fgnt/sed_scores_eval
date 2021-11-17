import numpy as np
from sed_scores_eval.utils.array_ops import get_first_index_where


def precision_recall_curve_from_intermediate_statistics(
        scores_intermediate_statistics
):
    if isinstance(scores_intermediate_statistics, dict):
        return {
            cls: precision_recall_curve_from_intermediate_statistics(
                scores_stats)
            for cls, scores_stats in scores_intermediate_statistics.items()
        }
    scores, stats = scores_intermediate_statistics
    assert isinstance(stats, dict), type(stats)
    assert all([key in stats for key in ['tps', 'fps', 'n_ref']]), stats.keys()
    p = stats['tps'] / np.maximum(stats['tps']+stats['fps'], 1)
    r = stats['tps'] / np.maximum(stats['n_ref'], 1)
    sort_idx = sorted(np.arange(len(r)).tolist(), key=lambda i: (r[i], p[i]))
    return (
        p[sort_idx], r[sort_idx], scores[sort_idx],
        {
            key: stat if np.isscalar(stat) else stat[sort_idx]
            for key, stat in stats.items()
         }
    )


def fscore_from_precision_recall(precision, recall, *, beta=1.):
    return (
        (1 + beta**2) * precision * recall
        / np.maximum(beta**2 * precision + recall, 1e-15)
    )


def fscore_curve_from_intermediate_statistics(
        scores_intermediate_statistics, beta=1.
):
    if isinstance(scores_intermediate_statistics, dict):
        f, p, r, scores, intermediate_stats = {}, {}, {}, {}, {}
        for cls, scores_stats in scores_intermediate_statistics.items():
            (
                f[cls], p[cls], r[cls], scores[cls], intermediate_stats[cls]
            ) = fscore_curve_from_intermediate_statistics(
                scores_stats, beta=beta,
            )
        return f, p, r, scores, intermediate_stats
    p, r, scores, intermediate_stats = precision_recall_curve_from_intermediate_statistics(
        scores_intermediate_statistics
    )
    sort_idx = np.argsort(scores)
    scores = scores[sort_idx]
    p = p[sort_idx]
    r = r[sort_idx]
    f_beta = fscore_from_precision_recall(p, r, beta=beta)
    intermediate_stats = {
        key: stat if np.isscalar(stat) else stat[sort_idx]
        for key, stat in intermediate_stats.items()
    }
    return f_beta, p, r, scores, intermediate_stats


def best_fscore_from_fscore_curve(
        fscore, precision, recall, scores, intermediate_statistics, *,
        min_precision=0., min_recall=0.,
):
    if isinstance(fscore, dict):
        f, p, r, thresholds, intermediate_stats = {}, {}, {}, {}, {}
        for cls in fscore.keys():
            (
                f[cls], p[cls], r[cls], thresholds[cls], intermediate_stats[cls]
            ) = best_fscore_from_fscore_curve(
                fscore[cls], precision[cls], recall[cls], scores[cls],
                intermediate_statistics=intermediate_statistics[cls],
                min_precision=min_precision, min_recall=min_recall,
            )
        return f, p, r, thresholds, intermediate_stats
    assert isinstance(fscore, np.ndarray), type(precision)
    assert isinstance(precision, np.ndarray), type(precision)
    assert isinstance(recall, np.ndarray), type(recall)
    assert isinstance(scores, np.ndarray), type(scores)
    assert fscore.ndim == 1, fscore.shape
    assert precision.ndim == 1, precision.shape
    assert recall.ndim == 1, recall.shape
    assert scores.ndim == 1, scores.shape
    fscore[precision < min_precision] = 0.
    fscore[recall < min_recall] = 0.
    best_idx = len(fscore) - 1 - np.argmax(fscore[::-1], axis=0)
    threshold = (
        (scores[best_idx] + scores[best_idx-1])/2 if best_idx > 0 else -np.inf
    )
    return (
        fscore[best_idx], precision[best_idx], recall[best_idx], threshold,
        {
            key: stat if np.isscalar(stat) else stat[best_idx]
            for key, stat in intermediate_statistics.items()
        }
    )


def best_fscore_from_intermediate_statistics(
        scores_intermediate_statistics, beta=1.,
        min_precision=0., min_recall=0.
):
    f, p, r, scores, intermediate_stats = fscore_curve_from_intermediate_statistics(
        scores_intermediate_statistics, beta=beta
    )
    f, p, r, threshold, intermediate_stats = best_fscore_from_fscore_curve(
        f, p, r, scores, intermediate_stats,
        min_precision=min_precision, min_recall=min_recall,
    )
    return f, p, r, threshold, intermediate_stats


def single_fscore_from_intermediate_statistics(
        scores_intermediate_statistics, threshold, *, beta=1., ):
    if isinstance(scores_intermediate_statistics, dict):
        f, p, r, intermediate_stats = {}, {}, {}, {}
        for cls, scores_stats in scores_intermediate_statistics.items():
            (
                f[cls], p[cls], r[cls], intermediate_stats[cls]
            ) = single_fscore_from_intermediate_statistics(
                scores_stats,
                threshold[cls] if isinstance(threshold, dict) else threshold,
                beta=beta,
            )
        return f, p, r, intermediate_stats
    pr_curve = precision_recall_curve_from_intermediate_statistics(
        scores_intermediate_statistics
    )
    p, r, scores, intermediate_stats = pr_curve
    return single_fscore_from_precision_recall_curve(
        p, r, scores, threshold, beta=beta,
        intermediate_statistics=intermediate_stats
    )


def single_fscore_from_precision_recall_curve(
        precisions, recalls, scores, threshold_of_interest, beta=1.,
        intermediate_statistics=None
):
    assert np.isscalar(threshold_of_interest), threshold_of_interest
    sort_idx = np.argsort(scores)
    scores = scores[sort_idx]
    idx = get_first_index_where(scores, 'gt', threshold_of_interest)
    p = precisions[sort_idx][idx]
    r = recalls[sort_idx][idx]
    f = fscore_from_precision_recall(p, r, beta=beta)
    if intermediate_statistics is None:
        return f, p, r
    else:
        return f, p, r, {
            key: stat if np.isscalar(stat) else stat[sort_idx][idx]
            for key, stat in intermediate_statistics.items()
        }
