import pytest
import numpy as np
import pandas as pd
from sed_scores_eval.segment_based import accumulated_intermediate_statistics


@pytest.mark.parametrize("seg_len", [5., 1.])
def test_triangular_scores(seg_len):

    detection_scores = np.array([.3,.4,.5,.6,.7,.8,.9,.8,.7,.6])
    timestamps = np.arange(11)
    scores = pd.DataFrame(
        np.array((timestamps[:-1], timestamps[1:], detection_scores)).T,
        columns=['onset', 'offset', 'a'],
    )
    change_point_scores, stats = accumulated_intermediate_statistics(
        scores={'1': scores},
        ground_truth={'1': [(2., 7., 'a')]},
        audio_durations={'1': len(detection_scores)},
        segment_length=seg_len,
    )[0]['a']
    if seg_len == 5.:
        expected_change_point_scores = [.7,.9,np.inf]
        expected_true_positives = [2,1,0]
        expected_false_positives = [0,0,0]
        expected_n_ref = 2
    elif seg_len == 1.:
        expected_change_point_scores = [.3,.4,.5,.6,.7,.8,.9,np.inf]
        expected_true_positives = [5,5,5,4,3,2,1,0]
        expected_false_positives = [5,4,3,3,2,1,0,0]
        expected_n_ref = 5
    else:
        raise NotImplementedError

    assert stats['n_ref'] == expected_n_ref, stats['n_ref']
    assert len(change_point_scores) == len(expected_change_point_scores), (
        change_point_scores, expected_change_point_scores)
    assert (change_point_scores == expected_change_point_scores).all(), (
        change_point_scores, expected_change_point_scores)
    assert (stats['tps'] == expected_true_positives).all(), (
        stats['tps'], expected_true_positives)
    assert (stats['fps'] == expected_false_positives).all(), (
        stats['fps'], expected_false_positives)
