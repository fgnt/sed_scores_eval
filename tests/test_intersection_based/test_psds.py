import pytest
import numpy as np
from scipy.interpolate import interp1d
from sed_scores_eval import intersection_based, io, package_dir


@pytest.mark.parametrize("dataset", ["eval"])
@pytest.mark.parametrize(
    "params",
    [
        {
            'dtc_threshold': .7,
            'gtc_threshold': .7,
            'cttc_threshold': None,
            'alpha_ct': .0,
            'alpha_st': 1 / np.sqrt(10 - 1),
            # choose alpha_st <= 1/sqrt(K-1), where K is the number of classes,
            # to ensure that the true PSD-ROC is always greater than an
            # approximation. With alpha_st > 1/sqrt(K-1) an increased TPR for
            # one of the classes may cause the effective TPR over all classes
            # to decrease.
        },
        {
            'dtc_threshold': .1,
            'gtc_threshold': .1,
            'cttc_threshold': .3,
            'alpha_ct': .5,
            'alpha_st': 1 / np.sqrt(10 - 1),
        },
    ]
)
def test_psds_vs_psds_eval(dataset, params):

    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    scores = io.read_sed_scores(test_data_dir / dataset / "scores")
    scores, ground_truth, audio_ids = io.parse_inputs(scores, test_data_dir / dataset / "ground_truth.tsv")
    audio_durations = io.parse_audio_durations(test_data_dir / dataset / "audio_durations.tsv", audio_ids=audio_ids)
    psds, psd_roc, single_class_psd_rocs = intersection_based.psds(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100., num_jobs=4, time_decimals=6,
    )
    thresholds = np.unique(np.concatenate([roc[2] for roc in single_class_psd_rocs.values()]))[:-1]
    (
        psds_ref, psd_roc_ref, single_class_psd_rocs_ref
    ) = intersection_based.reference.approximate_psds(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        thresholds=thresholds - 1e-15,
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100.,
    )
    assert 1e-3 > (psds - psds_ref) >= 0., (psds, psds_ref)

    assert_roc_geq_roc_ref(psd_roc, psd_roc_ref, 'psd-roc', upper_bound=2e-2)
    for event_class in single_class_psd_rocs:
        assert_roc_geq_roc_ref(
            single_class_psd_rocs[event_class],
            single_class_psd_rocs_ref[event_class],
            event_class,
            upper_bound=2e-2,
        )


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
