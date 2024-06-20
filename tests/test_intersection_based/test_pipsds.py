import pytest
import numpy as np
from functools import partial
from sed_scores_eval.base_modules.postprocessing import medfilt
from sed_scores_eval.base_modules.bootstrap import confidence_interval
from sed_scores_eval.utils.test import assert_roc_geq_roc_ref
from sed_scores_eval import io, package_dir
from sed_scores_eval import intersection_based


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
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_median_filter_independent_psds(dataset, params, num_jobs):
    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    median_filter_lengths_in_sec = np.arange(11)/5
    median_filter_lengths_in_sec[0] = 0.02

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
    (
        pipsds, single_class_pipsds, pi_psd_roc, single_class_pi_psd_rocs,
        psds_rocs, single_class_psd_rocs,
    ) = intersection_based.median_filter_independent_psds(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        median_filter_lengths_in_sec=median_filter_lengths_in_sec,
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100., time_decimals=6,
        num_jobs=num_jobs,
    )
    psd_roc_diff = np.array(psds_rocs[0]) - np.array(psd_roc_ref)
    assert (psd_roc_diff < 1e-9).all(), psd_roc_diff.max()
    assert pipsds >= psds_ref, (pipsds, psds_ref)
    print(pipsds, psds_ref)
    assert_roc_geq_roc_ref(pi_psd_roc, psd_roc_ref, 'pi-psd-roc')
    for class_name, pipsds_c in single_class_pipsds.items():
        psd_roc_diff = np.abs(np.array(single_class_psd_rocs[0][class_name][:2]) - np.array(single_class_psd_rocs_ref[class_name][:2]))
        assert (psd_roc_diff <= 1e-9).all(), (psd_roc_diff.max(), class_name)
        assert pipsds_c >= single_class_psds_ref[class_name], (pipsds_c, single_class_psds_ref[class_name], class_name)
        assert_roc_geq_roc_ref(single_class_pi_psd_rocs[class_name], single_class_psd_rocs_ref[class_name], class_name)


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
def test_bootstrapped_median_filter_independent_psds(dataset, params, num_jobs):
    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    median_filter_lengths_in_sec = np.arange(11)/5
    median_filter_lengths_in_sec[0] = 0.02

    scores = io.read_sed_scores(test_data_dir / dataset / "scores")
    scores, ground_truth, audio_ids = io.parse_inputs(scores, test_data_dir / dataset / "ground_truth.tsv")
    audio_durations = io.parse_audio_durations(test_data_dir / dataset / "audio_durations.tsv", audio_ids=audio_ids)
    (
        pipsds, single_class_pipsds, pi_psd_roc, single_class_pi_psd_rocs,
        psds_rocs, single_class_psd_rocs,
    ) = intersection_based.bootstrapped_median_filter_independent_psds(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        median_filter_lengths_in_sec=median_filter_lengths_in_sec,
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100., time_decimals=6,
        num_jobs=num_jobs, n_bootstrap_samples=20,
    )
    (psds_mean, psds_low, psds_high) = confidence_interval(pipsds)
    assert psds_low < psds_mean < psds_high, (psds_low, psds_mean, psds_high)
    print(psds_low, psds_mean, psds_high)
    single_class_psds_intervals = confidence_interval(single_class_pipsds)
    for class_name, (mean, low, high) in single_class_psds_intervals.items():
        assert low < mean < high, (low, mean, high)
    (
        pipsds, single_class_pipsds, pi_psd_roc, single_class_pi_psd_rocs,
        psds_rocs, single_class_psd_rocs,
    ) = intersection_based.median_filter_independent_psds(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        median_filter_lengths_in_sec=median_filter_lengths_in_sec,
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100., time_decimals=6,
        num_jobs=num_jobs,
    )
    assert psds_low < pipsds < psds_high, (psds_low, pipsds, psds_high)
    print(pipsds)
    for class_name, (mean, low, high) in single_class_psds_intervals.items():
        assert low < single_class_pipsds[class_name] < high, (low, single_class_pipsds[class_name], high)


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
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_bootstrapped_median_filter_independent_psds_prefiltered(dataset, params, num_jobs):
    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    median_filter_lengths_in_sec = np.arange(11)/5
    median_filter_lengths_in_sec[0] = 0.02

    scores = io.read_sed_scores(test_data_dir / dataset / "scores")
    scores, ground_truth, audio_ids = io.parse_inputs(scores, test_data_dir / dataset / "ground_truth.tsv")
    audio_durations = io.parse_audio_durations(test_data_dir / dataset / "audio_durations.tsv", audio_ids=audio_ids)
    (
        pipsds, single_class_pipsds, pi_psd_roc, single_class_pi_psd_rocs,
        psds_rocs, single_class_psd_rocs,
    ) = intersection_based.bootstrapped_median_filter_independent_psds(
        scores=scores,
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        median_filter_lengths_in_sec=median_filter_lengths_in_sec,
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100., time_decimals=6,
        num_jobs=num_jobs, n_bootstrap_samples=20,
    )

    postprocessing_functions = [
        partial(medfilt, filter_length_in_sec=filter_length_in_sec)
        for filter_length_in_sec in median_filter_lengths_in_sec
    ]
    (
        pipsds_prefiltered, single_class_pipsds_prefiltered,
        pi_psd_roc_prefiltered, single_class_pi_psd_rocs_prefiltered,
        psds_rocs_prefiltered, single_class_psd_rocs_prefiltered,
    ) = intersection_based.bootstrapped_median_filter_independent_psds(
        scores=None,
        deltas=[
            intersection_based.deltas_postprocessing(
                scores,
                ground_truth=ground_truth,
                postprocessing_fn=postprocessing_fn,
                dtc_threshold=params['dtc_threshold'],
                gtc_threshold=params['gtc_threshold'],
                cttc_threshold=params['cttc_threshold'],
            )
            for postprocessing_fn in postprocessing_functions
        ],
        ground_truth=ground_truth,
        audio_durations=audio_durations,
        median_filter_lengths_in_sec=median_filter_lengths_in_sec,
        dtc_threshold=params['dtc_threshold'],
        gtc_threshold=params['gtc_threshold'],
        cttc_threshold=params['cttc_threshold'],
        alpha_ct=params['alpha_ct'], alpha_st=params['alpha_st'],
        unit_of_time='hour', max_efpr=100., time_decimals=6,
        num_jobs=num_jobs, n_bootstrap_samples=20,
    )
    assert (np.array(pipsds_prefiltered) == np.array(pipsds)).all()
