import pytest
import numpy as np
from scipy.interpolate import interp1d
from sed_scores_eval import intersection_based, io, package_dir


@pytest.mark.parametrize("dataset", ["validation", "eval"])
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

    psds, psd_roc, single_class_psd_rocs = intersection_based.psds(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        audio_durations=test_data_dir / dataset / "audio_durations.tsv",
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100., num_jobs=8, time_decimals=6,
    )

    score_transform = test_data_dir / 'validation' / 'score_transform.tsv'
    if not score_transform.exists():
        io.write_score_transform(
            scores=test_data_dir / 'validation' / "scores",
            ground_truth=test_data_dir / 'validation' / "ground_truth.tsv",
            filepath=test_data_dir / 'validation' / 'score_transform.tsv'
        )

    (
        psds_approx, psd_roc_approx, single_class_psd_rocs_approx
    ) = intersection_based.reference.approximate_psds(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        audio_durations=test_data_dir / dataset / "audio_durations.tsv",
        thresholds=np.linspace(0.001, 0.999, 500),
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100.,
        score_transform=score_transform,
    )
    assert 1e-2 > (psds - psds_approx) >= 0., (psds, psds_approx)

    def assert_true_roc_geq_approx_roc(roc_true, roc_approx, class_name=''):
        tpr_true, fpr_true, *_ = roc_true
        fpr_true = np.round(fpr_true, 6)
        tpr_approx, fpr_approx, *_ = roc_approx
        fpr_approx = np.round(fpr_approx, 6)
        tpr_true = interp1d(
            fpr_true, tpr_true, kind='previous',
            bounds_error=False, fill_value=(0, tpr_true[-1])
        )(fpr_approx)
        assert (tpr_true >= tpr_approx).all(), (
            class_name,
            np.sum(tpr_true < tpr_approx),
            len(tpr_true),
            (tpr_approx - tpr_true).max(),
        )

    assert_true_roc_geq_approx_roc(psd_roc, psd_roc_approx, 'psd-roc')
    for event_class in single_class_psd_rocs:
        assert_true_roc_geq_approx_roc(
            single_class_psd_rocs[event_class],
            single_class_psd_rocs_approx[event_class],
            event_class,
        )
