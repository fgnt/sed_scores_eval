# distutils: language = c++
#cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector

cdef int searchsorted(double [:] sorted_list, double value, int init_idx):
    """Search position in a sorted list where value should be inserted.

    Args:
        sorted_list:
        value:
        init_idx: an index within the list from where to start searching

    Returns:

    """
    cdef int l = len(sorted_list)
    cdef int current_idx = min(init_idx, l-1)
    if sorted_list[current_idx] >= value:
        while current_idx > 0 and sorted_list[current_idx-1] >= value:
            current_idx -= 1
    else:
        while current_idx < l and sorted_list[current_idx] < value:
            current_idx += 1
    return current_idx

#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)
def cy_medfilt(scores_in, timestamps_in, filter_length_in_sec=None, time_decimals=6):
    """

    Args:
        scores_in (2d np.ndarray): scores array with shape TxK for a clip with T scores segments and K classes
        timestamps_in: onset timestamps for each frame plus one more
            timestamp which is the final offset time.
        filter_length_in_sec: filter length in seconds
        time_decimals: the decimal precision to use w.r.t. timestamps.

    Returns:

    """

    assert filter_length_in_sec is not None, "You must provide filter_length_in_sec"

    if np.isscalar(filter_length_in_sec):
        if filter_length_in_sec == 0.:
            return scores_in, timestamps_in
        filter_length_in_sec = np.array(scores_in.shape[1] * [filter_length_in_sec], dtype=np.float64)

    scores_in = np.asanyarray(scores_in, dtype=np.float64)
    timestamps_in = np.asanyarray(timestamps_in, dtype=np.float64)
    assert (timestamps_in[1:] > timestamps_in[:-1]).all(), np.min(timestamps_in[1:] - timestamps_in[:-1])
    filter_length_in_sec = np.asanyarray(filter_length_in_sec, dtype=np.float64)
    cdef double [:,:] overlap_breakpoint_timestamps = np.array([
        np.sort(
            np.round(np.concatenate((
                timestamps_in - fil_len / 2,
                timestamps_in + fil_len / 2,
            )), decimals=time_decimals + 2)
        )
        for fil_len in filter_length_in_sec
    ])
    # get breakpoints where the filter moves into or out of a segment

    cdef double [:] filter_length = filter_length_in_sec

    # Idea of median filter computation:
    # For a given time step, we compute the median as the center of a stacked
    # bar, where each segment, that overlaps with the filter at that time step,
    # is represented by a bar with height equal to the overlap with the filter.
    # Those bars are than stacked ordered by there corresponding score values
    # and the median value within the filter is than given as the center of the
    # stacked bar. This is done for all breakpoints and when there is a change
    # of median value between two breakpoints, the exact time point of median
    # value change can be computed by exploiting that between two breakpoints
    # the overlap/height of bars changes linearly. (The overlap/height of bar of
    # the segment the filter is moving out of, decreases linearly, whereas the
    # overlap/height of bar of the segment the filter is moving into,
    # increases linearly.)

    cdef int num_segments = scores_in.shape[0] + 2
    cdef int num_classes = scores_in.shape[1]
    cdef int num_breakpoints = overlap_breakpoint_timestamps.shape[1]
    cdef double [:,:] scores = np.concatenate((
        np.full_like(scores_in[:1], -np.inf),
        scores_in,
        np.full_like(scores_in[:1], -np.inf),
    ))
    cdef double [:] timestamps = np.concatenate(([-np.inf], timestamps_in, [np.inf]))
    cdef double eps = 10**(-time_decimals-1)

    cdef:
        int i, k, j, ranking_idx
        double t, t_diff, t_cp, current_cumsum, prev_cumsum, filter_length_k

    # for each class we track the current median score, the median bar lower
    # edge, which is the lower edge of the current median bar within the stacked
    # bar, and the bar upper edge, which is the upper edge of the current median
    # bar within the stacked bar.

    # track change_points where any of the classes change their median values
    cdef vector[double] change_point_timestamps = []
    cdef vector[int] change_point_class_labels = []
    cdef vector[double] change_point_class_scores = []
    cdef int relevant_segments_onset = 0
    cdef int relevant_segments_offset = 1
    cdef int num_relevant_segments = 1
    cdef int [:] ranking_indices = np.zeros(num_segments, dtype=np.int32)
    cdef double current_median_score = -np.inf
    cdef double current_median_score_range_low = 0
    cdef double current_median_score_range_up
    cdef double [:] scores_k
    cdef double [:] overlap_breakpoint_timestamps_k
    cdef double [:] relevant_scores = np.zeros(num_segments)
    cdef double [:] relevant_overlaps = np.zeros(num_segments)

    for k in range(num_classes):
        filter_length_k = filter_length[k]
        scores_k = scores[:, k]
        if filter_length_k == 0.:
            for i in range(1, num_segments):
                change_point_timestamps.push_back(timestamps[i])
                change_point_class_labels.push_back(k)
                change_point_class_scores.push_back(scores_k[i])
            continue
        current_median_score = -np.inf
        current_median_score_range_low = 0
        current_median_score_range_up = filter_length_k
        overlap_breakpoint_timestamps_k = overlap_breakpoint_timestamps[k]
        relevant_segments_onset = 0
        relevant_segments_offset = 1
        num_relevant_segments = 1
        relevant_scores[0] = -np.inf
        relevant_overlaps[0] = filter_length_k
        for i in range(1, num_breakpoints):
            t = overlap_breakpoint_timestamps_k[i]  # current breakpoint
            # get time difference to previous breakpoint, i.e., the time the filter
            # moved since last breakpoint
            t_diff = t - overlap_breakpoint_timestamps_k[i-1]
            if t_diff < eps:
                continue
            # get relevant segments range, i.e., segments the filter overlapped with
            # while moving from previous to current breakpoint and compute overlap
            # at current breakpoint
            if timestamps[relevant_segments_onset+1] < (t-filter_length_k/2 - eps):
                ranking_idx = searchsorted(
                    relevant_scores[:num_relevant_segments],
                    scores_k[relevant_segments_onset],
                    ranking_indices[relevant_segments_onset],
                )
                assert relevant_scores[ranking_idx] == scores_k[relevant_segments_onset]
                if relevant_overlaps[ranking_idx] < eps:
                    num_relevant_segments -= 1
                    for j in range(ranking_idx, num_relevant_segments):
                        relevant_scores[j] = relevant_scores[j+1]
                        relevant_overlaps[j] = relevant_overlaps[j+1]
                relevant_segments_onset += 1
            ranking_idx = searchsorted(
                relevant_scores[:num_relevant_segments],
                scores_k[relevant_segments_onset],
                ranking_indices[relevant_segments_onset],
            )
            assert relevant_scores[ranking_idx] == scores_k[relevant_segments_onset]
            relevant_overlaps[ranking_idx] -= t_diff
            ranking_indices[relevant_segments_onset] = ranking_idx
            if timestamps[relevant_segments_offset] < (t + filter_length_k/2 - eps):
                relevant_segments_offset += 1
                ranking_idx = searchsorted(
                    relevant_scores[:num_relevant_segments],
                    scores_k[relevant_segments_offset-1],
                    0,
                )
                ranking_indices[relevant_segments_offset - 1] = ranking_idx
                if ranking_idx == num_relevant_segments or relevant_scores[ranking_idx] != scores_k[relevant_segments_offset-1]:
                    num_relevant_segments += 1
                    for j in range(num_relevant_segments-1, ranking_idx, -1):
                        relevant_scores[j] = relevant_scores[j-1]
                        relevant_overlaps[j] = relevant_overlaps[j-1]
                    relevant_scores[ranking_idx] = scores_k[relevant_segments_offset-1]
                    relevant_overlaps[ranking_idx] = 0.
            else:
                ranking_idx = searchsorted(
                    relevant_scores[:num_relevant_segments],
                    scores_k[relevant_segments_offset-1],
                    ranking_indices[relevant_segments_offset - 1],
                )
            relevant_overlaps[ranking_idx] += t_diff
            if relevant_segments_offset == (relevant_segments_onset + 1):
                # if filter window lies completely within one segment than nothing changes
                continue
            #  update range of current median bar
            if scores_k[relevant_segments_onset] < current_median_score:
                current_median_score_range_low -= t_diff
                current_median_score_range_up -= t_diff
            elif scores_k[relevant_segments_onset] == current_median_score:
                current_median_score_range_up -= t_diff
            if scores_k[relevant_segments_offset-1] < current_median_score:
                current_median_score_range_low += t_diff
                current_median_score_range_up += t_diff
            elif scores_k[relevant_segments_offset-1] == current_median_score:
                current_median_score_range_up += t_diff
            # if center of stacked bar (filter_length_k / 2) is not within current
            # median bar range anymore, the median has changed while filter was
            # moving to the current breakpoint and we need to compute the exact
            # time point where the median value changed.
            if current_median_score_range_up < (filter_length_k/2 - eps):
                # change point
                current_cumsum = 0
                j = 0
                while current_cumsum < (filter_length_k/2 - eps):
                    prev_cumsum = current_cumsum
                    current_cumsum = current_cumsum + relevant_overlaps[j]
                    if current_cumsum > (current_median_score_range_up + eps):
                        t_cp = t - (filter_length_k/2 - prev_cumsum)
                        assert (relevant_scores[j] >= current_median_score), (relevant_scores[j], current_median_score, i, k, t)
                        if relevant_scores[j] > current_median_score:
                            change_point_timestamps.push_back(t_cp)
                            change_point_class_labels.push_back(k)
                            change_point_class_scores.push_back(relevant_scores[j])
                        if current_cumsum >= (filter_length_k/2 - eps):
                            current_median_score = relevant_scores[j]
                            current_median_score_range_up = current_cumsum
                            current_median_score_range_low = prev_cumsum
                    j += 1
            if current_median_score_range_low >= (filter_length_k/2 - eps):
                # change point
                current_cumsum = 0
                j = num_relevant_segments-1
                while current_cumsum <= (filter_length_k/2 - eps):
                    prev_cumsum = current_cumsum
                    current_cumsum = current_cumsum + relevant_overlaps[j]
                    if current_cumsum > (filter_length_k - current_median_score_range_up + eps):
                        t_cp = t - (filter_length_k/2 - prev_cumsum)
                        # assert (relevant_scores[j] <= current_median_scores), (relevant_scores[j], current_median_scores, i, k)
                        if relevant_scores[j] < current_median_score:
                            change_point_timestamps.push_back(t_cp)
                            change_point_class_labels.push_back(k)
                            change_point_class_scores.push_back(relevant_scores[j])
                        if current_cumsum >= (filter_length_k/2 - eps):
                            current_median_score = relevant_scores[j]
                            current_median_score_range_up = filter_length_k - prev_cumsum
                            current_median_score_range_low = filter_length_k - current_cumsum
                    j -= 1

    # wrap things up
    change_point_timestamps = np.round(np.array(change_point_timestamps), time_decimals)

    sort_idx = np.argsort(change_point_timestamps)
    change_point_timestamps_sorted = np.asarray(change_point_timestamps)[sort_idx]
    change_point_class_labels_sorted = np.asarray(change_point_class_labels)[sort_idx]
    change_point_class_scores_sorted = np.asarray(change_point_class_scores)[sort_idx]
    scores_filtered = []
    timestamps_filtered = []
    current_timestamp = -np.inf
    current_score_vec = scores.shape[1]*[-np.inf]
    for i, t in enumerate(change_point_timestamps_sorted):
        if t > (current_timestamp + eps):
            timestamps_filtered.append(t)
            scores_filtered.append(current_score_vec)
            current_score_vec = [*current_score_vec]
            current_timestamp = t
        current_score_vec[change_point_class_labels_sorted[i]] = change_point_class_scores_sorted[i]

    if len(scores_filtered) == 0:
        return np.full(scores_in.shape[1], -np.inf)[None], timestamps_in[[0,-1]]
    return np.array(scores_filtered)[1:], np.round(np.array(timestamps_filtered)+eps, time_decimals)
