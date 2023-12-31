import pytest
import numpy as np
import pandas as pd
from sed_scores_eval.intersection_based import accumulated_intermediate_statistics


@pytest.mark.parametrize("t_step", [.2, 1.])
def test_paper_example(t_step):
    detection_scores = np.array([.3,.3,.4,.5,.6,.7,.8,.7,.6])
    timestamps = np.arange(10)*t_step
    scores = pd.DataFrame(
        np.array((
            timestamps[:-1], timestamps[1:],
            detection_scores, np.zeros_like(detection_scores)
        )).T,
        columns=['onset', 'offset', 'a', 'b'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores},
        ground_truth={
            '1': [(2.*t_step, 6.*t_step, 'a'), (6.*t_step, 8.*t_step, 'b')]},
        dtc_threshold=.5, gtc_threshold=.5, cttc_threshold=.5,
    )[0]['a']
    expected_change_point_scores = [.3,.5,.6,.8, np.inf]
    expected_true_positives = [0,1,0,0,0]
    expected_false_positives = [1,0,1,1,0]
    expected_cross_triggers = [0,0,0,1,0]

    assert stats['n_ref'] == 1, stats['n_ref']
    assert np.abs(stats['t_ref'] - 4.*t_step) < 1e-12, (stats['t_ref'], 4.*t_step)
    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (
        (stats['tps'] == expected_true_positives).all()
    ), (stats['tps'], expected_true_positives)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)
    assert stats['cts'].keys() == {'b'}, stats['cts']
    assert stats['cts']['b'].shape == (len(change_point_scores),), (
        stats['cts']['b'], expected_cross_triggers)
    assert (
        (stats['cts']['b'] == expected_cross_triggers).all()
    ), (stats['cts']['b'], expected_cross_triggers)


@pytest.mark.parametrize("t_step", [.2, 1.])
def test_two_other_events(t_step):
    """

    Args:
        t_step: width of a frame/score in seconds

    Returns:

    """
    detection_scores = np.array([.3,.3,.4,.5,.6,.7,.8,.7,.6])
    timestamps = np.arange(10)*t_step
    scores = pd.DataFrame(
        np.array((
            timestamps[:-1], timestamps[1:],
            detection_scores, np.zeros_like(detection_scores),
            np.zeros_like(detection_scores)
        )).T,
        columns=['onset', 'offset', 'a', 'b', 'c'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores},
        ground_truth={
            '1': [
                (2.*t_step, 6.*t_step, 'a'),
                (6.*t_step, 8.*t_step, 'b'),
                (6.*t_step, 8.*t_step, 'c'),
            ],
        },
        dtc_threshold=.5, gtc_threshold=.5, cttc_threshold=.5,
    )[0]['a']
    expected_change_point_scores = [.3,.5,.6,.8, np.inf]
    expected_true_positives = [0,1,0,0,0]
    expected_false_positives = [1,0,1,1,0]
    expected_cross_triggers = [0,0,0,1,0]

    assert stats['n_ref'] == 1, stats['n_ref']
    assert np.abs(stats['t_ref'] - 4.*t_step) < 1e-12, (stats['t_ref'], 4.*t_step)
    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (
        (stats['tps'] == expected_true_positives).all()
    ), (stats['tps'], expected_true_positives)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)

    assert stats['cts'].keys() == {'b', 'c'}, stats['cts']
    assert stats['cts']['b'].shape == (len(change_point_scores),), (
        stats['cts']['b'], expected_cross_triggers)
    assert stats['cts']['c'].shape == (len(change_point_scores),), (
        stats['cts']['c'], expected_cross_triggers)
    assert (
        (stats['cts']['b'] == expected_cross_triggers).all()
    ), (stats['cts']['b'], expected_cross_triggers)
    assert (
        (stats['cts']['c'] == expected_cross_triggers).all()
    ), (stats['cts']['c'], expected_cross_triggers)


@pytest.mark.parametrize("t_step", [.2, 1.])
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_accumulated_statistics(t_step, num_jobs):
    """

    Args:
        t_step: width of a frame/score in seconds

    Returns:

    """
    detection_scores = np.array([.3,.3,.4,.5,.6,.7,.8,.7,.6])
    timestamps = np.arange(10)*t_step
    scores = pd.DataFrame(
        np.array((
            timestamps[:-1], timestamps[1:],
            detection_scores, np.zeros_like(detection_scores),
        )).T,
        columns=['onset', 'offset', 'a', 'b'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores, '2': scores},
        ground_truth={
            '1': [(2.*t_step, 6.*t_step, 'a'), (6.*t_step, 8.*t_step, 'b'),],
            '2': [(2.*t_step, 6.*t_step, 'a'), (6.*t_step, 8.*t_step, 'b'),]
        },
        dtc_threshold=.5, gtc_threshold=.5, cttc_threshold=.5,
        num_jobs=num_jobs,
    )[0]['a']
    expected_change_point_scores = [.3,.5,.6,.8, np.inf]
    expected_true_positives = [0,2,0,0,0]
    expected_false_positives = [2,0,2,2,0]
    expected_cross_triggers = [0,0,0,2,0]

    assert stats['n_ref'] == 2, stats['n_ref']
    assert np.abs(stats['t_ref'] - 2*4.*t_step) < 1e-12, (stats['t_ref'], 4.*t_step)
    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (
        (stats['tps'] == expected_true_positives).all()
    ), (stats['tps'], expected_true_positives)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)

    assert stats['cts'].keys() == {'b'}, stats['cts']
    assert stats['cts']['b'].shape == (len(change_point_scores),), (
        stats['cts']['b'], expected_cross_triggers)
    assert (
        (stats['cts']['b'] == expected_cross_triggers).all()
    ), (stats['cts']['b'], expected_cross_triggers)


@pytest.mark.parametrize("dtc_threshold", [.5, .6])
def test_event_offset_beyond_file_offset(dtc_threshold):
    detection_scores = np.concatenate((np.arange(5), 1+np.arange(3)[::-1]))
    detection_scores = np.concatenate((
        detection_scores, detection_scores[:-1][::-1]
    ))
    timestamps = np.arange(16)
    scores = pd.DataFrame(
        np.array((
            timestamps[:-1], timestamps[1:],
            detection_scores, np.zeros_like(detection_scores),
        )).T,
        columns=['onset', 'offset', 'a', 'b'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores},
        ground_truth={'1': [(7.5, 16., 'a'), (0., 7.5, 'b'),],},
        dtc_threshold=dtc_threshold, gtc_threshold=.5,
        cttc_threshold=dtc_threshold,
    )[0]['a']
    expected_change_point_scores = [1,2,4,np.inf]
    if dtc_threshold <= .5:
        expected_true_positives = [1,1,0,0]
        expected_false_positives = [0,1,1,0]
        expected_cross_triggers = [0,1,1,0]
    else:
        expected_true_positives = [0,1,0,0]
        expected_false_positives = [1,1,1,0]
        expected_cross_triggers = [0,1,1,0]

    assert stats['n_ref'] == 1, stats['n_ref']
    assert np.abs(stats['t_ref'] - 8.5) < 1e-12, stats['t_ref']
    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (
        (stats['tps'] == expected_true_positives).all()
    ), (stats['tps'], expected_true_positives)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)
    assert stats['cts'].keys() == {'b'}, stats['cts']
    assert stats['cts']['b'].shape == (len(change_point_scores),), (
        stats['cts']['b'], expected_cross_triggers)
    assert (
        (stats['cts']['b'] == expected_cross_triggers).all()
    ), (stats['cts']['b'], expected_cross_triggers)


@pytest.mark.parametrize("dtc_threshold", [.4, .45, .5, .7])
def test_non_overlapping_events(dtc_threshold):
    detection_scores = np.concatenate((
        np.zeros(4), 0.5*np.ones(7), np.ones(3), 0.5*np.ones(7),
        0.1*np.ones(8),
        0.5*np.ones(7), np.ones(3), 0.5*np.ones(7), np.zeros(4)
    ))
    timestamps = np.arange(51) / 10  # 100 ms steps
    scores = pd.DataFrame(
        np.array((
            timestamps[:-1], timestamps[1:],
            detection_scores, np.zeros_like(detection_scores),
            np.zeros_like(detection_scores),
        )).T,
        columns=['onset', 'offset', 'a', 'b', 'c'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores},
        ground_truth={
            '1': [
                (0.5, 1.0, 'a'), (1.0, 1.5, 'b'), (1.5, 2.0, 'a'),
                (3.0, 3.5, 'a'), (3.5, 4.0, 'c'), (4.0, 4.5, 'a'),
            ],
        },
        dtc_threshold=dtc_threshold, gtc_threshold=.5, cttc_threshold=.5,
    )[0]['a']
    if dtc_threshold <= 0.4:
        expected_change_point_scores = [.5, 1., np.inf]
        expected_true_positives = [4,0,0]
        expected_false_positives = [0,2,0]
        expected_cross_triggers = [[0,1,0],[0,1,0]]
    elif dtc_threshold <= (2./4.2):
        expected_change_point_scores = [0., .5, 1., np.inf]
        expected_true_positives = [0,4,0,0]
        expected_false_positives = [1,0,2,0]
        expected_cross_triggers = [[0,0,1,0],[0,0,1,0]]
    elif dtc_threshold <= (2./3.4):
        expected_change_point_scores = [.1, .5, 1., np.inf]
        expected_true_positives = [0,4,0,0]
        expected_false_positives = [1,0,2,0]
        expected_cross_triggers = [[0,0,1,0],[0,0,1,0]]
    else:
        expected_change_point_scores = [.1, .5, 1., np.inf]
        expected_true_positives = [0,0,0,0]
        expected_false_positives = [1,2,2,0]
        expected_cross_triggers = [[0,0,1,0],[0,0,1,0]]

    assert stats['n_ref'] == 4, stats['n_ref']
    assert np.abs(stats['t_ref'] - 2.) < 1e-12, stats['t_ref']
    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (
        (stats['tps'] == expected_true_positives).all()
    ), (stats['tps'], expected_true_positives)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)

    assert stats['cts'].keys() == {'b', 'c'}, stats['cts']
    assert stats['cts']['b'].shape == (len(change_point_scores),), (
        stats['cts']['b'], expected_cross_triggers)
    assert stats['cts']['c'].shape == (len(change_point_scores),), (
        stats['cts']['c'], expected_cross_triggers)
    assert (
        (stats['cts']['b'] == expected_cross_triggers).all()
    ), (stats['cts']['b'], expected_cross_triggers)
    assert (
        (stats['cts']['c'] == expected_cross_triggers).all()
    ), (stats['cts']['c'], expected_cross_triggers)


@pytest.mark.parametrize("dtc_threshold", [.3, .5])
def test_overlapping_events(dtc_threshold):
    detection_scores = np.array([1, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2])
    detection_scores = np.concatenate((detection_scores, [1], detection_scores[::-1]))
    timestamps = np.arange(26) / 10  # 100 ms steps
    scores = pd.DataFrame(
        np.array((
            timestamps[:-1], timestamps[1:],
            detection_scores, np.zeros_like(detection_scores)
        )).T,
        columns=['onset', 'offset', 'a', 'b'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores},
        ground_truth={
            '1': [(.2, 1.1, 'a'), (.2, 1.1, 'b'), (1.4, 2.3, 'b')]},
        dtc_threshold=dtc_threshold, gtc_threshold=.5,
        cttc_threshold=dtc_threshold,
    )[0]['a']
    expected_change_point_scores = [0.,1.,2.,3.,np.inf]
    if dtc_threshold <= .3:
        expected_true_positives = [1,1,0,0,0]
        expected_false_positives = [0,3,5,1,0]
        expected_cross_triggers = [0,3,1,1,0]
    else:
        expected_true_positives = [0,1,0,0,0]
        expected_false_positives = [1,4,5,1,0]
        expected_cross_triggers = [1,1,1,1,0]

    assert stats['n_ref'] == 1, stats['n_ref']
    assert np.abs(stats['t_ref'] - .9) < 1e-12, stats['t_ref']
    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (
        (stats['tps'] == expected_true_positives).all()
    ), (stats['tps'], expected_true_positives)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)

    assert stats['cts'].keys() == {'b'}, stats['cts']
    assert stats['cts']['b'].shape == (len(change_point_scores),), (
        stats['cts']['b'], expected_cross_triggers)
    assert (
        (stats['cts']['b'] == expected_cross_triggers).all()
    ), (stats['cts']['b'], expected_cross_triggers)
