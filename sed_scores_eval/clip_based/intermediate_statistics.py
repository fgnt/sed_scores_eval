import numpy as np
import multiprocessing
from sed_scores_eval.base_modules.io import parse_inputs
from sed_scores_eval.utils.scores import validate_score_dataframe


def intermediate_statistics(scores, ground_truth, *, num_jobs=1):
    """

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            tags for each audio clip or a file path from where the ground truth
            can be loaded.
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns:

    """
    if not isinstance(num_jobs, int) or num_jobs < 1:
        raise ValueError(
            f'num_jobs has to be an integer greater or equal to 1 but '
            f'{num_jobs} was given.'
        )
    scores, ground_truth, audio_ids = parse_inputs(
        scores, ground_truth, tagging=True)
    _, event_classes = validate_score_dataframe(scores[audio_ids[0]])

    if num_jobs == 1:
        clip_scores, clip_targets = _worker(audio_ids, scores, ground_truth, event_classes)
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
                    shard, scores, ground_truth, event_classes, queue
                ),
                daemon=True,
            )
            for shard in shards
        ]
        try:
            for p in processes:
                p.start()
            clip_scores, clip_targets = None, None
            count = 0
            while count < len(shards):
                clip_scores_i, clip_targets_i = queue.get()
                if clip_scores is None:
                    clip_scores = clip_scores_i
                    clip_targets = clip_targets_i
                else:
                    for class_name in clip_scores:
                        clip_scores[class_name].extend(clip_scores_i[class_name])
                        clip_targets[class_name].extend(clip_targets_i[class_name])
                count += 1
        finally:
            for p in processes:
                p.terminate()
    stats = {}
    for class_name in event_classes:
        clip_scores[class_name] = np.array(clip_scores[class_name]+[np.inf])
        sort_idx = np.argsort(clip_scores[class_name])
        clip_scores[class_name] = clip_scores[class_name][sort_idx]
        clip_targets[class_name] = np.array(
            clip_targets[class_name]+[0])[sort_idx]
        tps = np.cumsum(clip_targets[class_name][::-1])[::-1]
        n_sys = np.arange(len(tps))[::-1]
        clip_scores[class_name], unique_idx = np.unique(
            clip_scores[class_name], return_index=True)
        n_ref = tps[0]
        fns = n_ref - tps
        tns = n_sys[0] - n_sys - fns
        stats[class_name] = {
            'tps': tps[unique_idx],
            'fps': n_sys[unique_idx] - tps[unique_idx],
            'tns': tns,
            'n_ref': n_ref,
        }
    return {
        class_name: (clip_scores[class_name], stats[class_name])
        for class_name in event_classes
    }


def _worker(audio_ids, scores, ground_truth, event_classes=None, output_queue=None):
    clip_scores = None
    clip_targets = None
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
        if clip_scores is None:
            clip_scores = {class_name: [] for class_name in event_classes}
            clip_targets = {class_name: [] for class_name in event_classes}
        for c, class_name in enumerate(event_classes):
            clip_scores[class_name].append(scores_k[c])
            clip_targets[class_name].append(class_name in gt_k)
    if output_queue is not None:
        output_queue.put((clip_scores, clip_targets))
    return clip_scores, clip_targets
