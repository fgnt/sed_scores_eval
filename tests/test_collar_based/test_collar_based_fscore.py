import pytest
import numpy as np
from sed_scores_eval import collar_based, io, package_dir
from sed_scores_eval.base_modules.precision_recall import fscore_from_sed_eval_metrics


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
def test_collar_based_fscore_vs_sed_eval(dataset, threshold, collar):
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
        num_jobs=8,
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
