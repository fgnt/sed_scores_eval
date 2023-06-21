import numpy as np
from scipy import signal
import pytest
import time
from sed_scores_eval.base_modules.cy_medfilt import cy_medfilt
from sed_scores_eval.base_modules.postprocessing import medfilt
from sed_scores_eval.base_modules.scores import create_score_dataframe
from sed_scores_eval import io, package_dir


def test_simple_1():
    scores = np.array([[0,0,1.,0,0],[0,0,1.,0,0]]).T
    timestamps = np.arange(6)/5
    scores_filt, timestamps_filt = cy_medfilt(scores, timestamps, .3)
    assert (scores_filt == np.array([[0.,1.,0.],[0.,1.,0.]]).T).all(), scores_filt
    assert (timestamps_filt == np.array([[0. , 0.4, 0.6, 1. ]])).all(), timestamps_filt


def test_simple_2():
    scores = np.array([[0,.5,1,1,0],[0,.5,1,1,0]]).T
    timestamps = np.arange(6)/5
    scores_filt, timestamps_filt = cy_medfilt(scores, timestamps, .5)
    assert (scores_filt == np.array([[0.,0.5,1.,0.],[0.,0.5,1.,0.]]).T).all(), scores_filt
    assert (timestamps_filt == np.array([[0., 0.2, 0.4, 0.8, 1. ]])).all(), timestamps_filt


def test_simple_3():
    scores = np.array([[0,.5,1,0,1],[0,.5,1,0,1]]).T
    timestamps = np.array([.0,.2,.4,.7,.8,1.])
    scores_filt, timestamps_filt = cy_medfilt(scores, timestamps, .6)
    assert (scores_filt == np.array([[0.,0.5,1.,0.],[0.,0.5,1.,0.]]).T).all(), scores_filt
    assert (timestamps_filt == np.array([[0., 0.2, 0.5, 0.9, 1. ]])).all(), timestamps_filt


@pytest.mark.parametrize("segment_length", [.02,.1/6])
@pytest.mark.parametrize("medfilt_length", [21,51,101,201])
def test_vs_scipy(segment_length, medfilt_length):
    scores = np.random.RandomState(0).rand(1000, 10)
    time_decimals = 6
    timestamps = np.arange(1001)*segment_length
    scores_filtered_ref = np.apply_along_axis(
        lambda m: signal.medfilt(m, medfilt_length),
        axis=0, arr=scores,
    )
    prev_scores = np.concatenate((
        np.zeros_like(scores_filtered_ref[:1]), scores_filtered_ref[:-1]
    ))
    change_points = np.argwhere((scores_filtered_ref != prev_scores).any(-1)).flatten()
    scores_filtered_ref = scores_filtered_ref[change_points]
    scores_filtered, timestamps_filtered = cy_medfilt(scores, timestamps, segment_length, time_decimals=time_decimals)
    assert scores_filtered.shape == scores.shape, (scores_filtered.shape, scores.shape)
    #assert (np.round(timestamps_filtered, time_decimals) == np.round(timestamps, time_decimals)).all()
    abs_diff = np.abs(scores_filtered - scores)
    assert (abs_diff < 1e-6).all(), (abs_diff.mean(), abs_diff.max(), (abs_diff >= 1e-6).sum())
    tic = time.time()
    scores_filtered, timestamps_filtered = cy_medfilt(
        scores, timestamps, medfilt_length * segment_length,
        time_decimals=6,
    )
    print(time.time() - tic)
    assert scores_filtered.shape == scores_filtered_ref.shape, (scores_filtered.shape, scores_filtered_ref.shape)
    abs_diff = np.abs(scores_filtered - scores_filtered_ref)
    assert (abs_diff < 1e-6).all(), (abs_diff.mean(), abs_diff.max(), (abs_diff >= 1e-6).sum())


@pytest.mark.parametrize("medfilt_length", [21,51,101,201])
def test_dataframes(medfilt_length):
    scores = np.random.RandomState(0).rand(1000, 10)
    timestamps = np.round(np.arange(0., 20.01, 0.02), 6)
    event_classes = list('abcdefghij')
    scores_df = create_score_dataframe(scores, timestamps, event_classes)
    tic = time.time()
    scores_filtered_ref, timestamps_filtered_ref = cy_medfilt(scores, timestamps, medfilt_length * 0.02)
    print(time.time() - tic)
    tic = time.time()
    scores_filtered = medfilt({'1': scores_df}, medfilt_length*0.02)['1'][event_classes].to_numpy()
    print(time.time() - tic)
    assert scores_filtered.shape == scores_filtered_ref.shape, (scores_filtered.shape, scores_filtered_ref.shape)
    abs_diff = np.abs(scores_filtered - scores_filtered_ref)
    assert (abs_diff < 1e-6).all(), (abs_diff.mean(), abs_diff.max(), (abs_diff >= 1e-6).sum())


@pytest.mark.parametrize("dataset", ["validation", "eval"])
def test_real_data(dataset):
    test_data_dir = package_dir / 'tests' / 'data'
    if not test_data_dir.exists():
        io.download_test_data()

    median_filter_length_in_sec = .02
    scores = io.read_sed_scores(test_data_dir / dataset / "scores")
    for audio_id, scores_i in medfilt(scores, median_filter_length_in_sec).items():
        scores_ref = scores[audio_id].to_numpy()[:, 2:]
        prev_scores = np.concatenate((
            np.zeros_like(scores_ref[:1]), scores_ref[:-1]
        ))
        change_points = np.argwhere((scores_ref != prev_scores).any(-1)).flatten()
        scores_ref = scores_ref[change_points]
        assert (scores_i.to_numpy()[:,2:] == scores_ref).all()
