import numpy as np
from sklearn.metrics import roc_auc_score
from sed_scores_eval import segment_based, utils


def test_segment_based_area_under_roc_vs_sklearn():
    segment_length = 2.
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    timestamps = np.arange(len(y_scores)+1) * segment_length
    auroc_sklearn = roc_auc_score(y_true, y_scores)
    gt = {
        '1': [(timestamps[idx], timestamps[idx+1], 'a') for idx, t in enumerate(y_true) if t]
    }
    scores = {
        '1': utils.create_score_dataframe(
            y_scores[..., None], timestamps=timestamps, event_classes=['a']
        )
    }
    audio_durations = {'1': timestamps[-1]}
    auroc, roc = segment_based.auroc(
        scores, gt, audio_durations, segment_length=segment_length,
    )

    assert auroc['a'] == auroc_sklearn, (auroc, auroc_sklearn)
