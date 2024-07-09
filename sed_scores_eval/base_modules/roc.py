import numpy as np
from sed_scores_eval.base_modules.curves import xsort, linear_auc


def roc_curve_from_intermediate_statistics(scores_intermediate_statistics):
    """compute ROC (TPR-FPR) curve from intermediate statistics

    Args:
        scores_intermediate_statistics ((dict of) tuple): tuple of scores array
            and dict of intermediate_statistics with the following key value
            pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'n_ref' (int): number of ground truth events
            If dict input is provided keys are expected to be class names with
            corresponding scores/intermediate_statistics as values.

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
    if isinstance(scores_intermediate_statistics, dict):
        roc_curves = {}
        for class_name, scores_stats in scores_intermediate_statistics.items():
            roc_curves[class_name] = roc_curve_from_intermediate_statistics(scores_stats)
        return roc_curves

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

    tpr = stats['tps'] / stats['n_ref']
    fpr = stats['fps'] / (stats['tns'] + stats['fps'])

    return xsort(tpr, fpr, scores, stats)


def auroc_from_intermediate_statistics(scores_intermediate_statistics, max_fpr=None, mcclish_correction=False):
    """compute area under ROC curve

    Args:
        scores_intermediate_statistics ((dict of) tuple): tuple of scores array
            and dict of intermediate_statistics with the following key value
            pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'n_ref' (int): number of ground truth events
            If dict input is provided keys are expected to be class names with
            corresponding scores/intermediate_statistics as values.
        max_fpr (float): maximum false positive rate up to which to compute partial auc
        mcclish_correction: whether to use mcclish correction to get result back
            into [0.5,1.] range when using max_fpr.

    Returns:
        auroc ((dict of) float): mean and class-wise area under ROC curve
        roc_curve: ROC (TPR-FPR) curve(s) as provided by
            `roc_curve_from_intermediate_statistics`

    """
    if isinstance(scores_intermediate_statistics, dict):
        auroc, roc_curves = {}, {}
        for class_name, scores_stats in scores_intermediate_statistics.items():
            auroc[class_name], roc_curves[class_name] = auroc_from_intermediate_statistics(
                scores_stats, max_fpr=max_fpr, mcclish_correction=mcclish_correction,
            )
        auroc['mean'] = np.mean([auroc[class_name] for class_name in auroc])
        return auroc, roc_curves
    roc_curve = roc_curve_from_intermediate_statistics(
        scores_intermediate_statistics
    )
    tpr, fpr, *_ = roc_curve
    auroc = linear_auc(tpr, fpr, max_x=max_fpr)
    if max_fpr is not None and mcclish_correction:
        min_area = 0.5 * max_fpr ** 2
        max_area = max_fpr
        auroc = 0.5 * (1 + (auroc - min_area) / (max_area - min_area))
    else:
        norm = 1 if max_fpr is None else max_fpr
        auroc = auroc/norm
    return auroc, roc_curve
