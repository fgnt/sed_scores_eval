import numpy as np
import pandas as pd
from sed_scores_eval.clip_based import accumulated_intermediate_statistics


def test_triangular_scores():
    detection_scores = np.array([.3,.4,.5,.6,.7,.8,.9,.8,.7,.6])
    timestamps = np.arange(11)
    scores_1 = pd.DataFrame(
        np.array((timestamps[:-1], timestamps[1:], detection_scores)).T,
        columns=['onset', 'offset', 'a'],
    )
    scores_2 = pd.DataFrame(
        np.array((
            timestamps[:-1], timestamps[1:],
            np.minimum(detection_scores, .4)
        )).T,
        columns=['onset', 'offset', 'a'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores_1, '2': scores_2},
        ground_truth={'1': ['a'], '2': []},
    )[0]['a']
    expected_change_point_scores = [.4,.9,np.inf]
    expected_true_positives = [1,1,0]
    expected_false_positives = [1,0,0]
    expected_n_ref = 1

    assert stats['n_ref'] == expected_n_ref, stats['n_ref']
    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (stats['tps'] == expected_true_positives).all(), (
        stats['tps'], expected_true_positives)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)
