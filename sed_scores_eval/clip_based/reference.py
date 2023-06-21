import numpy as np
from sed_scores_eval.base_modules.scores import validate_score_dataframe
from sed_scores_eval.base_modules.io import parse_inputs


def metrics(scores, ground_truth, threshold):
    """Reference metrics implementation using the sed_eval package
    (https://tut-arg.github.io/sed_eval/), which, however, does not allow to
    compute metrics at various operating points simultaneously.
    This function is primarily used for testing purposes.

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            tags for each audio clip or a file path from where the ground truth
            can be loaded.
        threshold ((dict of) float): threshold that is to be evaluated.

    Returns (sed_eval.audio_tag.AudioTaggingMetrics): audio tagging metrics

    """
    import sed_eval
    import dcase_util

    scores, ground_truth, keys = parse_inputs(scores, ground_truth, tagging=True)
    _, event_classes = validate_score_dataframe(scores[keys[0]])
    if isinstance(threshold, dict):
        threshold = [threshold[event_class] for event_class in event_classes]
        if not all([np.isscalar(thr) for thr in threshold]):
            raise ValueError('All values of thresholds dict must be scalars')
        threshold = np.asanyarray(threshold)
    elif not np.isscalar(threshold):
        raise ValueError(
            f'threshold must be (dict of) scalar(s) but {type(threshold)} '
            f'was given.'
        )

    estimated_tag_list = []
    for audio_id in keys:
        scores_k = scores[audio_id]
        validate_score_dataframe(scores_k, event_classes=event_classes)
        scores_k = scores_k[event_classes].to_numpy().max(0)
        estimated_tags = scores_k > threshold
        estimated_tags = [
            class_name for c, class_name in enumerate(event_classes)
            if estimated_tags[c]
        ]
        estimated_tag_list.append({
            'filename': f'{audio_id}.wav',
            'tags': ','.join(estimated_tags),
        })

    estimated_tag_list = dcase_util.containers.MetaDataContainer(
        estimated_tag_list)
    reference_tag_list = dcase_util.containers.MetaDataContainer([
        {
            'filename': f'{audio_id}.wav',
            'tags': ','.join(tags),
        }
        for audio_id, tags in ground_truth.items()
    ])

    tag_evaluator = sed_eval.audio_tag.AudioTaggingMetrics(
        tags=reference_tag_list.unique_tags
    )
    tag_evaluator.evaluate(
        reference_tag_list=reference_tag_list,
        estimated_tag_list=estimated_tag_list,
    )
    return tag_evaluator
