import pytest
import numpy as np
import pandas as pd
from sed_scores_eval.collar_based import accumulated_intermediate_statistics


@pytest.mark.parametrize("t_step", [.2, 1.])
def test_paper_example(t_step):
    detection_scores = np.array([.3,.4,.5,.6,.7,.6,.5,.4,.3])
    timestamps = np.arange(10)*t_step
    scores = pd.DataFrame(
        np.array((timestamps[:-1], timestamps[1:], detection_scores)).T,
        columns=['onset', 'offset', 'a'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores},
        ground_truth={'1': [(2.*t_step, 7.*t_step, 'a')]},
        onset_collar=t_step, offset_collar=t_step,
    )[0]['a']
    expected_change_point_scores = [.3,.6,.7,np.inf]
    expected_true_positives = [0,1,0,0]
    expected_false_positives = [1,0,1,0]

    assert stats['n_ref'] == 1, stats['n_ref']
    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (stats['tps'] == expected_true_positives).all(), (
        stats['tps'], expected_true_positives)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)


@pytest.mark.parametrize("t_step", [.2, 1.])
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_accumulated_statistics(t_step, num_jobs):
    detection_scores = np.array([.3,.4,.5,.6,.7,.6,.5,.4,.3])
    timestamps = np.arange(10)*t_step
    scores = pd.DataFrame(
        np.array((timestamps[:-1], timestamps[1:], detection_scores)).T,
        columns=['onset', 'offset', 'a'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores, '2': scores},
        ground_truth={
            '1': [(2.*t_step, 7.*t_step, 'a')],
            '2': [(2.*t_step, 7.*t_step, 'a')]
        },
        onset_collar=t_step, offset_collar=t_step,
        num_jobs=num_jobs,
    )[0]['a']
    expected_change_point_scores = [.3,.6,.7,np.inf]
    expected_true_positives = [0,2,0,0]
    expected_false_positives = [2,0,2,0]

    assert stats['n_ref'] == 2, stats['n_ref']
    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)
    assert (stats['tps'] == expected_true_positives).all(), (
        stats['tps'], expected_true_positives)


@pytest.mark.parametrize("collar", [.2,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
def test_two_events(collar):
    detection_scores = np.concatenate(([0,3], np.arange(5), np.arange(4)[::-1]))
    timestamps = np.arange(12)
    scores = pd.DataFrame(
        np.array((timestamps[:-1], timestamps[1:], detection_scores)).T,
        columns=['onset', 'offset', 'a'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores},
        ground_truth={'1': [(0.,2.5,'a'), (4.5,8.5,'a')]},
        onset_collar=collar, offset_collar=collar,
    )[0]['a']
    expected_change_point_scores = [0,3,4,np.inf]
    expected_num_detections = [1,2,1,0]
    if collar < .5:
        expected_true_positives = [0,0,0,0]
    elif collar < 1.5:
        expected_change_point_scores = [0,1,3,4,np.inf]
        expected_num_detections = [1,2,2,1,0]
        expected_true_positives = [0,1,2,0,0]
    elif collar < 4.5:
        expected_true_positives = [0,2,1,0]
    else:
        expected_true_positives = [1,2,+1,0]
    expected_false_positives = np.array(expected_num_detections) - expected_true_positives

    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (stats['tps'] == expected_true_positives).all(), (
        stats['tps'], expected_true_positives)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)


@pytest.mark.parametrize("collar", [.2,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
def test_zeros(collar):
    detection_scores = np.zeros(11)
    timestamps = np.arange(12)
    scores = pd.DataFrame(
        np.array((timestamps[:-1], timestamps[1:], detection_scores)).T,
        columns=['onset', 'offset', 'a'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores},
        ground_truth={'1': [(0.,2.5,'a'), (4.5,8.5,'a')]},
        onset_collar=collar, offset_collar=collar,
    )[0]['a']
    expected_change_point_scores = [0.,np.inf]
    if collar < 4.5:
        expected_true_positives = [0,0]
        expected_false_positives = [1,0]
    else:
        expected_true_positives = [1,0]
        expected_false_positives = [0,0]

    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (stats['tps'] == expected_true_positives).all(), (
        stats['tps'], expected_true_positives)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)


@pytest.mark.parametrize("collar", [0.,1.,2.,3.,4.,5.,6.,7.,8.])
def test_spawn_at_first_score(collar):
    detection_scores = np.arange(11)[::-1]
    timestamps = np.arange(12)
    scores = pd.DataFrame(
        np.array((timestamps[:-1], timestamps[1:], detection_scores)).T,
        columns=['onset', 'offset', 'a'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores},
        ground_truth={'1': [(0.,1.,'a')]},
        onset_collar=collar, offset_collar=collar,
    )[0]['a']
    expected_change_point_scores = [9.-collar, 10., np.inf]
    expected_true_positives = [0, 1, 0]
    expected_false_positives = [1, 0, 0]

    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (stats['tps'] == expected_true_positives).all(), (
        stats['tps'], expected_true_positives)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)
