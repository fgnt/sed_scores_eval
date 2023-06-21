import numpy as np
import multiprocessing
from sed_scores_eval.base_modules.scores import validate_score_dataframe
from sed_scores_eval.base_modules import statistics
from sed_scores_eval.base_modules.io import parse_inputs


def intermediate_statistics_deltas(
        scores, ground_truth, *, num_jobs=1,
):
    if not isinstance(num_jobs, int) or num_jobs < 1:
        raise ValueError(
            f'num_jobs has to be an integer greater or equal to 1 but '
            f'{num_jobs} was given.'
        )
    scores, ground_truth, audio_ids = parse_inputs(
        scores, ground_truth, tagging=True)
    _, event_classes = validate_score_dataframe(scores[audio_ids[0]])

    if num_jobs == 1:
        deltas = _worker(audio_ids, scores, ground_truth, event_classes)
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
                    shard, scores, ground_truth, event_classes, queue
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


def accumulated_intermediate_statistics(scores, ground_truth, *, deltas=None, num_jobs=1):
    """

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            tags for each audio clip or a file path from where the ground truth
            can be loaded.
        deltas (dict of dicts of tuples): Must be deltas as returned by
            `accumulated_intermediate_statistics_from_deltas`. If not provided,
            deltas are computed within this function. Providing deltas is useful
            if deltas are used repeatedly as, e.g., with bootstrapped evaluation,
            to save computing time.
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns:

    """
    if deltas is None:
        scores, ground_truth, audio_ids = parse_inputs(scores, ground_truth, tagging=True)
        deltas = intermediate_statistics_deltas(
            scores=scores, ground_truth=ground_truth, num_jobs=num_jobs,
        )
    else:
        audio_ids = list(deltas.keys())
    return accumulated_intermediate_statistics_from_deltas(deltas), audio_ids


def _worker(audio_ids, scores, ground_truth, event_classes=None, output_queue=None):
    deltas = {}
    for audio_id in audio_ids:
        scores_k = scores[audio_id]
        timestamps, _ = validate_score_dataframe(
            scores_k, event_classes=event_classes)
        scores_k = scores_k[event_classes].to_numpy().max(0)
        gt_k = ground_truth[audio_id]
        if not all([
            class_name in event_classes for class_name in gt_k
        ]):
            unknown_tags = [
                class_name for class_name in gt_k
                if class_name not in event_classes
            ]
            raise ValueError(
                f'ground truth contains unknown tags. Unknown tags: '
                f'{unknown_tags}; Known tags: {event_classes};'
            )
        deltas[audio_id] = {}
        for c, class_name in enumerate(event_classes):
            deltas[audio_id][class_name] = ([scores_k[c]], {'tps': [class_name in gt_k], 'fps': [class_name not in gt_k]})
    if output_queue is not None:
        output_queue.put(deltas)
    return deltas
