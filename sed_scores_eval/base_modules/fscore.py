import numpy as np
from sed_scores_eval.utils.array_ops import get_first_index_where


def fscore_from_intermediate_statistics(
        intermediate_statistics, threshold, *, beta=1., ):
    pr_curve = precision_recall_curve_from_intermediate_statistics(
        intermediate_statistics
    )
    if isinstance(pr_curve, dict):
        f, p, r, intermediate_stats = {}, {}, {}, {}
        for cls, (pi, ri, scores_i, stats_i) in pr_curve.items():
            (
                f[cls], p[cls], r[cls], intermediate_stats[cls]
            ) = single_fscore_from_precision_recall_curve(
                pi, ri, scores_i,
                threshold[cls] if isinstance(threshold, dict) else threshold,
                beta=beta, intermediate_statistics=stats_i
            )
        return f, p, r, intermediate_stats
    p, r, scores, intermediate_stats = pr_curve
    return single_fscore_from_precision_recall_curve(
        p, r, scores, threshold, beta=beta,
        intermediate_statistics=intermediate_stats
    )


def best_fscore_from_intermediate_statistics(
        intermediate_statistics, *, beta=1., min_precision=0., min_recall=0.,
):
    pr_curve = precision_recall_curve_from_intermediate_statistics(
        intermediate_statistics
    )
    if isinstance(pr_curve, dict):
        f, p, r, thr, intermediate_stats = {}, {}, {}, {}, {}
        for cls, (pi, ri, scores_i, stats_i) in pr_curve.items():
            (
                f[cls], p[cls], r[cls], thr[cls], intermediate_stats[cls]
            ) = best_fscore_from_precision_recall_curve(
                pi, ri, scores_i, beta=beta,
                min_precision=min_precision, min_recall=min_recall,
                intermediate_statistics=stats_i,
            )
        return f, p, r, thr, intermediate_stats
    p, r, scores, intermediate_stats = pr_curve
    return best_fscore_from_precision_recall_curve(
        p, r, scores, beta=beta,
        min_precision=min_precision, min_recall=min_recall,
        intermediate_statistics=intermediate_stats
    )


def precision_recall_curve_from_intermediate_statistics(
        intermediate_statistics
):
    if isinstance(intermediate_statistics, dict):
        return {
            cls: precision_recall_curve_from_intermediate_statistics(
                scores_stats)
            for cls, scores_stats in intermediate_statistics.items()
        }
    scores, stats = intermediate_statistics
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


def best_fscore_from_precision_recall_curve(
        precision, recall, scores, *, beta=1., min_precision=0., min_recall=0.,
        intermediate_statistics=None
):
    assert isinstance(precision, np.ndarray), type(precision)
    assert isinstance(recall, np.ndarray), type(recall)
    assert isinstance(scores, np.ndarray), type(scores)
    assert precision.ndim == 1, precision.shape
    assert recall.ndim == 1, recall.shape
    assert scores.ndim == 1, scores.shape
    sort_idx = np.argsort(scores)
    scores = scores[sort_idx]
    precision = precision[sort_idx]
    recall = recall[sort_idx]
    f_beta = fscore_from_precision_recall(precision, recall, beta=beta)
    f_beta[precision < min_precision] = 0.
    f_beta[recall < min_recall] = 0.
    best_idx = len(f_beta) - 1 - np.argmax(f_beta[::-1], axis=0)
    threshold = (
        (scores[best_idx] + scores[best_idx-1])/2 if best_idx > 0 else -np.inf
    )
    if intermediate_statistics is None:
        return (
            f_beta[best_idx], precision[best_idx], recall[best_idx], threshold
        )
    else:
        return (
            f_beta[best_idx], precision[best_idx], recall[best_idx], threshold,
            {
                key: stat if np.isscalar(stat) else stat[sort_idx[best_idx]]
                for key, stat in intermediate_statistics.items()
            }
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
