import numpy as np
import multiprocessing
from sed_scores_eval.utils.array_ops import get_first_index_where
from sed_scores_eval.base_modules.scores import validate_score_dataframe
from sed_scores_eval.base_modules import statistics
from sed_scores_eval.base_modules.io import parse_inputs, parse_audio_durations
from sed_scores_eval.base_modules.ground_truth import multi_label_to_single_label_ground_truths


def intermediate_statistics_deltas(
        scores, ground_truth, audio_durations, *,
        segment_length=1., time_decimals=6, num_jobs=1,
):
    if not isinstance(num_jobs, int) or num_jobs < 1:
        raise ValueError(
            f'num_jobs has to be an integer greater or equal to 1 but '
            f'{num_jobs} was given.'
        )
    scores, ground_truth, audio_ids = parse_inputs(scores, ground_truth)
    if audio_durations is not None:
        audio_durations = parse_audio_durations(audio_durations, audio_ids=audio_ids)

    _, event_classes = validate_score_dataframe(scores[audio_ids[0]])
    single_label_ground_truths = multi_label_to_single_label_ground_truths(
        ground_truth, event_classes)

    if num_jobs == 1:
        deltas = _worker(
            audio_ids, scores, single_label_ground_truths, audio_durations,
            segment_length,  event_classes, time_decimals,
        )
    else:
        deltas = {}
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
                    shard, scores, single_label_ground_truths, audio_durations,
                    segment_length,  event_classes, time_decimals, queue
                ),
                daemon=True,
            )
            for shard in shards
        ]
        try:
            for p in processes:
                p.start()
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
    multi_label_statistics = statistics.accumulated_intermediate_statistics_from_deltas(deltas)
    for _, stats_cls in multi_label_statistics.values():
        stats_cls['n_sys'] = stats_cls['tps'] + stats_cls['fps']
        stats_cls['n_ref'] = stats_cls['tps'][0]
        fns = stats_cls['n_ref'] - stats_cls['tps']
        stats_cls['tns'] = stats_cls['n_sys'][0] - stats_cls['n_sys'] - fns
    return multi_label_statistics


def accumulated_intermediate_statistics(
        scores, ground_truth, audio_durations, *, deltas=None,
        segment_length=1., time_decimals=6, num_jobs=1,
):
    """Cascade of `intermediate_statistics_deltas` and `accumulated_intermediate_statistics_from_deltas`.

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        audio_durations: The duration of each audio file in the evaluation set.
        segment_length: the segment length of the segments that are to be
            evaluated.
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high detected or ground truth events that have
            onsets or offsets right on a segment boundary may swap over to the
            adjacent segment because of small deviations due to limited
            floating point precision.
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns:

    """
    if deltas is None:
        scores, ground_truth, audio_ids = parse_inputs(scores, ground_truth)
        deltas = intermediate_statistics_deltas(
            scores=scores, ground_truth=ground_truth,
            audio_durations=audio_durations,
            segment_length=segment_length, time_decimals=time_decimals,
            num_jobs=num_jobs,
        )
    else:
        audio_ids = list(deltas.keys())
    return accumulated_intermediate_statistics_from_deltas(deltas), audio_ids


def _worker(
        audio_ids, scores, single_label_ground_truths, audio_durations,
        segment_length=1., event_classes=None, time_decimals=6,
        output_queue=None,
):
    deltas = {}
    for audio_id in audio_ids:
        scores_k = scores[audio_id]
        timestamps, _ = validate_score_dataframe(
            scores_k, event_classes=event_classes)
        timestamps = np.round(timestamps, time_decimals)
        scores_k = scores_k[event_classes].to_numpy()
        if audio_durations is None:
            duration = max(
                [timestamps[-1]] + [
                    t_off for class_name in event_classes
                    for t_on, t_off, *_ in single_label_ground_truths[class_name][audio_id]
                ]
            )
        else:
            duration = audio_durations[audio_id]
        n_segments = int(np.ceil(duration / segment_length))
        segment_boundaries = np.round(
            np.arange(n_segments+1) * segment_length,
            time_decimals
        )
        segment_onsets = segment_boundaries[:-1]
        segment_offsets = segment_boundaries[1:]
        tp_deltas = {}
        for class_name in event_classes:
            gt = single_label_ground_truths[class_name][audio_id]
            if len(gt) == 0:
                tp_deltas[class_name] = np.zeros(n_segments, dtype=bool)
            else:
                tp_deltas[class_name] = np.any([
                    (segment_onsets < gt_offset)
                    * (segment_offsets > gt_onset)
                    * (segment_offsets > segment_onsets)
                    for gt_onset, gt_offset in
                    single_label_ground_truths[class_name][audio_id]
                ], axis=0)
        cp_scores = {class_name: [] for class_name in event_classes}
        for i in range(n_segments):
            idx_on = get_first_index_where(
                timestamps, "gt", segment_onsets[i]) - 1
            idx_on = max(idx_on, 0)
            idx_off = get_first_index_where(
                timestamps, "geq", segment_offsets[i])
            idx_off = min(idx_off, len(timestamps)-1)
            if idx_off <= idx_on:
                scores_ki = np.zeros(scores_k.shape[-1])
            else:
                scores_ki = np.max(scores_k[idx_on:idx_off], axis=0)
            for c, class_name in enumerate(event_classes):
                cp_scores[class_name].append(scores_ki[c])
        if audio_id not in deltas:
            deltas[audio_id] = {}
        for class_name in tp_deltas:
            deltas[audio_id][class_name] = (np.array(cp_scores[class_name]), {'tps': tp_deltas[class_name], 'fps': 1-tp_deltas[class_name]})
    if output_queue is not None:
        output_queue.put(deltas)
    return deltas
