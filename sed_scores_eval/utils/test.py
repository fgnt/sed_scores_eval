import numpy as np
from scipy.interpolate import interp1d


def assert_roc_geq_roc_ref(roc, roc_ref, class_name='', upper_bound=None):
    tpr, fpr, *_ = roc
    fpr = np.round(fpr, 6)
    tpr_ref, fpr_ref, *_ = roc_ref
    fpr_ref = np.round(fpr_ref, 6)
    tpr = interp1d(
        fpr, tpr, kind='previous',
        bounds_error=False, fill_value=(0, tpr[-1])
    )(fpr_ref)
    assert (tpr >= tpr_ref).all(), (
        class_name,
        np.sum(tpr < tpr_ref),
        len(tpr),
        (tpr_ref - tpr).max(),
    )
    if upper_bound is not None:
        assert ((tpr - tpr_ref) < upper_bound).all(), (
            class_name,
            np.sum((tpr - tpr_ref) >= upper_bound),
            len(tpr),
            (tpr - tpr_ref).max(),
        )
