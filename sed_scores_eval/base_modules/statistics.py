import numpy as np
import multiprocessing
from sed_scores_eval.utils.nested import flatten, deflatten
from sed_scores_eval.base_modules.scores import validate_score_dataframe
from sed_scores_eval.base_modules.ground_truth import multi_label_to_single_label_ground_truths
from sed_scores_eval.base_modules.cy_detection import onset_offset_curves
from sed_scores_eval.base_modules.io import parse_inputs


def intermediate_statistics_deltas(
        scores, ground_truth, intermediate_statistics_fn, *,
        acceleration_fn=None, num_jobs=1,
        **intermediate_statistics_fn_kwargs
):
    """Core function of this package. It computes the deltas of intermediate
    statistics for single audio files as described in our paper [1] for all
    files in the dataset. Note that this function assumes intermediate
    statistics to be 0 for a decision threshold of infinity, i.e., when no event
    is detected. So the intermediate statistics have to be defined accordingly.

    [1] J.Ebbers, R.Serizel, and R.Haeb-Umbach
    "Threshold-Independent Evaluation of Sound Event Detection Scores",
    in Proc. IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP),
    2022

    Args:
        scores (dict, str, pathlib.Path): dict of multi-label SED score
            DataFrames (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        intermediate_statistics_fn (callable): a function returning a dict of
            intermediate statistics for a single target class and a single
            audio file by taking (at least) the following key word arguments
            (See collar_based.intermediate_statistics.statistics_fn or
            intersection_based.intermediate_statistics.statistics_fn for
            examples):
              detection_onset_times (np.ndarray): (B, M) onset times of
                detected target class events with M being the number of
                detected target class events, and B being an independent
                dimension.
              detection_offset_times (np.ndarray): (B, M) offset times of
                detected target class events with M being the number of
                detected target class events, and B being an independent
                dimension. Note that it may include offset times which are
                equal to the corresponding onset time, which indicates that the
                event is inactive at that specific position b along the
                independent axis and must not be counted as a detection.
              target_onset_times (1d np.ndarray): onset times of target class
                ground truth events.
              target_offset_times (1d np.ndarray): offset times of target class
                ground truth events.
              other_onset_times (list of 1d np.ndarrays): onset times of other
                class ground truth events
              other_offset_times (list of 1d np.ndarrays): offset times of
                other class ground truth events
        acceleration_fn (callable): a function returning a reduced set of
            change point candidates and/or directly the change point scores with
            corresponding intermediate statistic deltas.
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.
        **intermediate_statistics_fn_kwargs: some other key word arguments for
            intermediate_statistics_fn, e.g., the collar in collar-based
            evaluation.

    Returns:
        deltas (dict of dicts of tuples): For each audio clip for each event class:
            change_point_scores (dict of dicts): 1d array of scores at which the
                intermediate statistics change, when the threshold falls below
                it, for that particular clip and class.
            dict of delta values: provides for each intermediate statistic a
                1d array of the delta (change), when the threshold falls below
                each of the change_point_scores.

    """
    if not isinstance(num_jobs, int) or num_jobs < 1:
        raise ValueError(
            f'num_jobs has to be an integer greater or equal to 1 but '
            f'{num_jobs} was given.'
        )
    scores, ground_truth, audio_ids = parse_inputs(scores, ground_truth)

    _, event_classes = validate_score_dataframe(scores[audio_ids[0]])
    single_label_ground_truths = multi_label_to_single_label_ground_truths(
        ground_truth, event_classes)

    if num_jobs == 1:
        deltas = _worker(
            audio_ids, scores, single_label_ground_truths,
            intermediate_statistics_fn, intermediate_statistics_fn_kwargs,
            acceleration_fn,
        )
    else:
        queue = multiprocessing.Queue()
        shard_size = int(np.ceil(len(audio_ids) / num_jobs))
        shards = [
            audio_ids[i*shard_size:(i+1)*shard_size] for i in range(num_jobs)
            if i*shard_size < len(audio_ids)
        ]
        processes = [
            multiprocessing.Process(
                target=_worker,
                args=(
                    shard, scores, single_label_ground_truths,
                    intermediate_statistics_fn,
                    intermediate_statistics_fn_kwargs,
                    acceleration_fn,
                    queue
                ),
                daemon=True,
            )
            for shard in shards
        ]
        try:
            for p in processes:
                p.start()
            deltas = {}
            count = 0
            while count < len(shards):
                deltas_i = queue.get()
                assert len(deltas.keys() & deltas_i.keys()) == 0, sorted(deltas.keys() & deltas_i.keys())
                deltas.update(deltas_i)
                count += 1
        finally:
            for p in processes:
                p.terminate()
    return deltas


def accumulated_intermediate_statistics_from_deltas(deltas):
    """
    Takes change_point_scores and deltas from `intermediate_statistics_deltas`.
    It then brings deltas from all audio clips in a single list sorted w.r.t.
    core values and computes intermediate statistics (over all clips) at various
    operating points by a cumulative sum over the deltas as described in our
    paper [1].

    Args:
        deltas (dict of dicts of tuples): For each audio clip for each event class:
            change_point_scores (dict of dicts): 1d array of scores at which the
                intermediate statistics change, when the threshold falls below
                it, for that particular clip and class.
            dict of delta values: provides for each intermediate statistic a
                1d array of the delta (change), when the threshold falls below
                each of the change_point_scores.

    Returns (dict of tuples): for each event class:
        - unique scores (1d np.ndarray) for which the intermediate statistics
            change when the threshold falls below it.
        - intermediate statistics (dict of 1d np.ndarray): dict of
            arrays of intermediate statistics for each of the scores.

    """
    audio_ids = list(deltas.keys())
    event_classes = list(deltas[audio_ids[0]].keys())
    acc_deltas = {
        class_name: ([], {key: [] for key in deltas[audio_ids[0]][class_name][1]})
        for class_name in event_classes
    }
    for audio_id in audio_ids:
        for class_name in event_classes:
            acc_deltas[class_name][0].append(deltas[audio_id][class_name][0])
            for key in acc_deltas[class_name][1]:
                acc_deltas[class_name][1][key].append(
                    deltas[audio_id][class_name][1][key])

    return {
        class_name: _intermediate_statistics_from_deltas(
            np.concatenate(acc_deltas[class_name][0]),
            {
                key: np.concatenate(acc_deltas[class_name][1][key])
                for key in acc_deltas[class_name][1]
            }
        )
        for class_name in event_classes
    }


def accumulated_intermediate_statistics(
        scores, ground_truth, intermediate_statistics_fn, *,
        acceleration_fn=None, num_jobs=1,
        **intermediate_statistics_fn_kwargs
):
    """Cascade of `intermediate_statistics_deltas` and `accumulated_intermediate_statistics_from_deltas`.

    Args:
        scores:
        ground_truth:
        intermediate_statistics_fn:
        acceleration_fn:
        num_jobs:
        **intermediate_statistics_fn_kwargs:

    Returns:

    """
    deltas = intermediate_statistics_deltas(
        scores, ground_truth, intermediate_statistics_fn,
        acceleration_fn=acceleration_fn, num_jobs=num_jobs,
        **intermediate_statistics_fn_kwargs
    )
    return accumulated_intermediate_statistics_from_deltas(deltas)


def _worker(
        audio_ids, scores, single_label_ground_truths,
        intermediate_statistics_fn, intermediate_statistics_fn_kwargs,
        acceleration_fn=None, output_queue=None
):
    deltas = {}
    _, event_classes = validate_score_dataframe(scores[audio_ids[0]])
    for audio_id in audio_ids:
        scores_for_key = scores[audio_id]
        timestamps, _ = validate_score_dataframe(
            scores_for_key, event_classes=event_classes)
        scores_for_key = scores_for_key[event_classes].to_numpy()
        deltas[audio_id] = {}
        gt_onset_times = {}
        gt_offset_times = {}
        for class_name in event_classes:
            gt = single_label_ground_truths[class_name][audio_id]
            if gt:
                current_onset_times, current_offset_times = np.array(gt).T
            else:
                current_onset_times = current_offset_times = np.empty(0)
            gt_onset_times[class_name] = current_onset_times
            gt_offset_times[class_name] = current_offset_times
        for c, class_name in enumerate(event_classes):
            target_onset_times = gt_onset_times[class_name]
            target_offset_times = gt_offset_times[class_name]
            other_onset_times = {ocls: onset_time for ocls, onset_time in gt_onset_times.items() if ocls != class_name}
            other_offset_times = {ocls: offset_time for ocls, offset_time in gt_offset_times.items() if ocls != class_name}
            if acceleration_fn is None:
                change_point_candidates = cp_scores_c = deltas_c = None
            else:
                change_point_candidates, cp_scores_c, deltas_c = acceleration_fn(
                    scores=scores_for_key[:, c], timestamps=timestamps,
                    target_onset_times=target_onset_times,
                    target_offset_times=target_offset_times,
                    other_onset_times=other_onset_times,
                    other_offset_times=other_offset_times,
                    **intermediate_statistics_fn_kwargs,
                )
                assert not (cp_scores_c is None) ^ (deltas_c is None)
            if cp_scores_c is None:
                (
                    unique_scores, detection_onset_times, detection_offset_times,
                ) = onset_offset_curves(
                    scores_for_key[:, c], timestamps, change_point_candidates
                )
                stats = intermediate_statistics_fn(
                    detection_onset_times=detection_onset_times,
                    detection_offset_times=detection_offset_times,
                    target_onset_times=target_onset_times,
                    target_offset_times=target_offset_times,
                    other_onset_times=other_onset_times,
                    other_offset_times=other_offset_times,
                    **intermediate_statistics_fn_kwargs,
                )
                cp_scores_c, deltas_c = _deltas_from_intermediate_statistics(
                    unique_scores, stats
                )
            if audio_id not in deltas:
                deltas[audio_id] = {}
            deltas[audio_id][class_name] = (cp_scores_c, deltas_c)
    if output_queue is not None:
        output_queue.put(deltas)
    return deltas


def _deltas_from_intermediate_statistics(scores, intermediate_stats):
    """compute deltas for intermediate statistics for single audio and single
    target class

    Args:
        scores (1d np.ndarray): single class SED scores from a single audio.
        intermediate_stats (dict of 1d np.ndarrays): dict of arrays of
            intermediate statistics for each of the scores

    Returns:
        change_point_scores (1d np.ndarray): array of scores for which the
            intermediate statistics change when the threshold falls below it.
        deltas (dict of 1d np.ndarrays): dict of arrays of the changes (deltas)
            in each intermediate statistic at each of the change point scores.

    """
    scores_unique, unique_idx = np.unique(scores, return_index=True)
    intermediate_stats = flatten(intermediate_stats, sep='\\')
    stat_keys = list(intermediate_stats.keys())
    stats = np.stack([
        intermediate_stats[key][unique_idx]
        for key in stat_keys
    ], axis=1)
    deltas = stats - np.concatenate((stats[1:], np.zeros_like(stats[:1])))

    # filter scores where nothing changes
    any_delta = np.abs(deltas).sum(1) > 0
    change_indices = np.argwhere(any_delta).flatten()
    change_point_scores = scores_unique[change_indices]
    deltas = {
        key: deltas_i[change_indices]
        for key, deltas_i in zip(stat_keys, deltas.T)
    }
    return change_point_scores, deltas


def _intermediate_statistics_from_deltas(change_point_scores, deltas):
    """sort and cumsum the deltas from all audio for each intermediate statistic

    Args:
        change_point_scores (1d np.ndarray): concatenation of single class SED scores from
            all audios.
        deltas (dict of 1d np.ndarrays): dict of concatenations of the changes
            (deltas) in each intermediate statistic at each of the scores.

    Returns:
        scores_unique:
        stats:
    """
    scores_unique, inverse_idx = np.unique(change_point_scores, return_inverse=True)
    b = len(scores_unique)
    scores_unique = np.concatenate((scores_unique, [np.inf]))
    stat_keys = list(deltas.keys())
    deltas = np.stack([deltas[key] for key in stat_keys], axis=1)
    deltas_unique = np.zeros((b, *deltas.shape[1:]))
    np.add.at(deltas_unique, inverse_idx, deltas)
    stats = np.concatenate((
        np.cumsum(deltas_unique[::-1], axis=0)[::-1],
        np.zeros_like(deltas_unique[:1])
    ))
    stats = deflatten({
        key: stats_i for key, stats_i in zip(stat_keys, stats.T)
    }, sep='\\')
    return scores_unique, stats
