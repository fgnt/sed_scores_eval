import numpy as np
from sklearn.metrics import average_precision_score
from sed_scores_eval.base_modules.scores import create_score_dataframe
from sed_scores_eval import clip_based


def test_clip_based_average_precision_vs_sklearn():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    ap_sklearn = average_precision_score(y_true, y_scores)
    gt = {
        str(idx): ['a'] if t else []
        for idx, t in enumerate(y_true)
    }
    scores = {
        str(idx): create_score_dataframe(
            s[None, None], timestamps=[0., 10.], event_classes=['a']
        )
        for idx, s in enumerate(y_scores)
    }
    ap, pr_curves = clip_based.average_precision(scores, gt)

    assert ap['a'] == ap_sklearn, (ap, ap_sklearn)
