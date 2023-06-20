import pytest
import numpy as np
from sed_scores_eval import collar_based, io, package_dir
from sed_scores_eval.base_modules.error_rate import error_rate_from_sed_eval_metrics


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
def test_collar_based_error_rate_vs_sed_eval(dataset, threshold, collar, num_jobs):
    offset_collar_rate = collar
    time_decimals = 30
    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    er, ir, dr, stats = collar_based.error_rate(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        threshold=threshold,
        onset_collar=collar, offset_collar=collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals, num_jobs=num_jobs,
    )
    sed_eval_metrics = collar_based.reference.metrics(
        scores=test_data_dir / dataset / "scores",
        ground_truth=test_data_dir / dataset / "ground_truth.tsv",
        threshold=threshold,
        collar=collar, offset_collar_rate=offset_collar_rate,
    )
    er_sed_eval, ir_sed_eval, dr_sed_eval = error_rate_from_sed_eval_metrics(
        sed_eval_metrics)
    for key in er.keys():
        np.testing.assert_almost_equal(er[key], er_sed_eval[key])
        np.testing.assert_almost_equal(ir[key], ir_sed_eval[key])
        np.testing.assert_almost_equal(dr[key], dr_sed_eval[key])
