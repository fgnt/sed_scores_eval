import pytest
import numpy as np
from sed_scores_eval import io, package_dir
from sed_scores_eval import intersection_based


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
@pytest.mark.parametrize("dtc_gtc_threshold", [(.1, .1), (.5, .5), (.7, .7), (.7, .5)])
def test_intersection_based_fscore_vs_psds_eval(dataset, threshold, dtc_gtc_threshold):
    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    dtc_threshold, gtc_threshold = dtc_gtc_threshold
    f, p, r, stats = intersection_based.fscore(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        threshold=threshold,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        time_decimals=30,
    )
    f_ref = intersection_based.reference.fscore(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        threshold=threshold,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
    )
    for key in f_ref.keys():
        np.testing.assert_almost_equal(f[key], f_ref[key])


@pytest.mark.parametrize("dataset", ["validation", "eval"])
@pytest.mark.parametrize("dtc_gtc_threshold", [(.1, .1), (.7, .7)])
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_intersection_based_best_fscore(dataset, dtc_gtc_threshold, num_jobs):
    dtc_threshold, gtc_threshold = dtc_gtc_threshold
    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    best_f, _, _, best_thresholds, _ = intersection_based.best_fscore(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        gtc_threshold=gtc_threshold, dtc_threshold=dtc_threshold,
        num_jobs=num_jobs,
    )
    f_ref, *_ = intersection_based.fscore(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        threshold=best_thresholds,
        gtc_threshold=gtc_threshold, dtc_threshold=dtc_threshold,
        num_jobs=num_jobs,
    )
    for key in f_ref.keys():
        assert abs(best_f[key] - f_ref[key]) < 1e-6, key
