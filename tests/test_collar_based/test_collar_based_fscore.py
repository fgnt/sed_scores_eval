import pytest
import numpy as np
from sed_scores_eval.base_modules.precision_recall import fscore_from_sed_eval_metrics
from sed_scores_eval.base_modules.bootstrap import confidence_interval
from sed_scores_eval import io, package_dir
from sed_scores_eval import collar_based


@pytest.mark.parametrize("dataset", ["validation", "eval"])
@pytest.mark.parametrize(
    "threshold",
    [
        .5,
        {
            'Alarm_bell_ringing': 0.9713,
            'Blender': 0.7718,
            'Cat': 0.8573,
            'Dishes': 0.9535,
            'Dog': 0.9495,
            'Electric_shaver_toothbrush': 0.9616,
            'Frying': 0.7752,
            'Running_water': 0.9555,
            'Speech': 0.8314,
            'Vacuum_cleaner': 0.6164
        }
    ]
)
@pytest.mark.parametrize("collar", [.2, .5])
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_collar_based_fscore_vs_sed_eval(dataset, threshold, collar, num_jobs):
    offset_collar_rate = collar
    time_decimals = 30
    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    f, p, r, stats = collar_based.fscore(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        threshold=threshold,
        onset_collar=collar, offset_collar=collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals,
        num_jobs=num_jobs,
    )
    sed_eval_metrics = collar_based.reference.metrics(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        threshold=threshold,
        collar=collar, offset_collar_rate=offset_collar_rate,
    )
    f_sed_eval, p_sed_eval, r_sed_eval = fscore_from_sed_eval_metrics(
        sed_eval_metrics)
    for key in f.keys():
        np.testing.assert_almost_equal(f[key], f_sed_eval[key])
        np.testing.assert_almost_equal(p[key], p_sed_eval[key])
        np.testing.assert_almost_equal(r[key], r_sed_eval[key])


@pytest.mark.parametrize("dataset", ["validation", "eval"])
@pytest.mark.parametrize(
    "threshold",
    [
        .5,
        {
            'Alarm_bell_ringing': 0.9713,
            'Blender': 0.7718,
            'Cat': 0.8573,
            'Dishes': 0.9535,
            'Dog': 0.9495,
            'Electric_shaver_toothbrush': 0.9616,
            'Frying': 0.7752,
            'Running_water': 0.9555,
            'Speech': 0.8314,
            'Vacuum_cleaner': 0.6164
        }
    ]
)
@pytest.mark.parametrize("collar", [.2, .5])
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_bootstrapped_collar_based_fscore(dataset, threshold, collar, num_jobs):
    offset_collar_rate = collar
    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    f, *_ = collar_based.bootstrapped_fscore(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        threshold=threshold,
        onset_collar=collar, offset_collar=collar,
        offset_collar_rate=offset_collar_rate,
        num_jobs=num_jobs, n_bootstrap_samples=20,
    )
    f_intervals = confidence_interval(f)
    for class_name, (f_mean, f_low, f_high) in f_intervals.items():
        assert f_low < f_mean < f_high, (f_low, f_mean, f_high)

    f, p, r, stats = collar_based.fscore(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        threshold=threshold,
        onset_collar=collar, offset_collar=collar,
        offset_collar_rate=offset_collar_rate,
        num_jobs=num_jobs,
    )
    for class_name, (f_mean, f_low, f_high) in f_intervals.items():
        assert f_low < f[class_name] < f_high, (f_low, f_mean, f_high)


@pytest.mark.parametrize("dataset", ["validation", "eval"])
@pytest.mark.parametrize("collar", [.2, .5])
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_collar_based_best_fscore(dataset, collar, num_jobs):
    offset_collar_rate = collar
    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    best_f, _, _, best_thresholds, _ = collar_based.best_fscore(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        onset_collar=collar, offset_collar=collar,
        offset_collar_rate=offset_collar_rate,
        num_jobs=num_jobs,
    )
    f_ref, *_ = collar_based.fscore(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        threshold=best_thresholds,
        onset_collar=collar, offset_collar=collar,
        offset_collar_rate=offset_collar_rate,
        num_jobs=num_jobs,
    )
    for key in f_ref.keys():
        assert abs(best_f[key] - f_ref[key]) < 1e-6, key
