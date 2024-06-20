import pytest
import numpy as np
from sed_scores_eval.base_modules.bootstrap import confidence_interval
from sed_scores_eval.utils.test import assert_roc_geq_roc_ref
from sed_scores_eval import io, package_dir
from sed_scores_eval import intersection_based
from sed_scores_eval.base_modules.scores import validate_score_dataframe, create_score_dataframe


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
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_psds_vs_psds_eval(dataset, params, num_jobs):

    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    scores = io.read_sed_scores(test_data_dir / dataset / "scores")
    scores, ground_truth, audio_ids = io.parse_inputs(scores, test_data_dir / dataset / "ground_truth.tsv")
    audio_durations = io.parse_audio_durations(test_data_dir / dataset / "audio_durations.tsv", audio_ids=audio_ids)
    psds, single_class_psds, psd_roc, single_class_psd_rocs = intersection_based.psds(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100., time_decimals=6, num_jobs=num_jobs,
    )
    thresholds = np.unique(np.concatenate([roc[2] for roc in single_class_psd_rocs.values()]))[:-1]
    (
        psds_ref, single_class_psds_ref, psd_roc_ref, single_class_psd_rocs_ref
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
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_psds_repeated_scores(dataset, params, num_jobs):

    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    scores = io.read_sed_scores(test_data_dir / dataset / "scores")
    scores, ground_truth, audio_ids = io.parse_inputs(scores, test_data_dir / dataset / "ground_truth.tsv")
    audio_durations = io.parse_audio_durations(test_data_dir / dataset / "audio_durations.tsv", audio_ids=audio_ids)
    psds_ref, single_class_psds_ref, psd_roc_ref, single_class_psd_rocs_ref = intersection_based.psds(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100., time_decimals=6, num_jobs=num_jobs,
    )
    def _repeat_scores(scores_df):
        timestamps, event_classes = validate_score_dataframe(scores_df)
        scores_arr = scores_df[event_classes].to_numpy()
        scores_arr = np.repeat(scores_arr, 2, axis=0)
        timestamps = np.concatenate((
            np.stack((timestamps[:-1], (timestamps[1:]+timestamps[:-1])/2), axis=-1).flatten(),
            timestamps[-1:]
        ))
        idx = (np.arange(1, len(timestamps)+1) % 4) > 0
        return create_score_dataframe(scores_arr[idx[:-1]], timestamps[idx], event_classes)
    scores_repeated = {key: _repeat_scores(scores_df) for key, scores_df in scores.items()}
    psds, single_class_psds, psd_roc, single_class_psd_rocs = intersection_based.psds(
        scores=scores_repeated,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100., time_decimals=6, num_jobs=num_jobs,
    )
    assert abs(psds - psds_ref) < 1e-6, (psds, psds_ref)



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
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_bootstrapped_psds(dataset, params, num_jobs):
    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    scores = io.read_sed_scores(test_data_dir / dataset / "scores")
    scores, ground_truth, audio_ids = io.parse_inputs(scores, test_data_dir / dataset / "ground_truth.tsv")
    audio_durations = io.parse_audio_durations(test_data_dir / dataset / "audio_durations.tsv", audio_ids=audio_ids)
    psds, single_class_psds, *_ = intersection_based.bootstrapped_psds(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100., time_decimals=6,
        num_jobs=num_jobs, n_bootstrap_samples=20,
    )
    (psds_mean, psds_low, psds_high) = confidence_interval(psds)
    assert psds_low < psds_mean < psds_high, (psds_low, psds_mean, psds_high)
    print(psds_low, psds_mean, psds_high)
    single_class_psds_intervals = confidence_interval(single_class_psds)
    for class_name, (mean, low, high) in single_class_psds_intervals.items():
        assert low < mean < high, (low, mean, high)
    psds, single_class_psds, psd_roc, single_class_psd_rocs = intersection_based.psds(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100., time_decimals=6, num_jobs=num_jobs,
    )
    assert psds_low < psds < psds_high, (psds_low, psds, psds_high)
    print(psds)
    for class_name, (mean, low, high) in single_class_psds_intervals.items():
        assert low < single_class_psds[class_name] < high, (low, single_class_psds[class_name], high)
