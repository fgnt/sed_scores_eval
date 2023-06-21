from pathlib import Path

import lazy_dataset
from functools import partial
from sed_scores_eval.base_modules.scores import validate_score_dataframe, create_score_dataframe
from sed_scores_eval.base_modules.cy_medfilt import cy_medfilt
from sed_scores_eval.base_modules.io import lazy_sed_scores_loader, write_sed_scores


def medfilt(scores, filter_length_in_sec, time_decimals=6, lazy=False, storage_dir=None):
    return apply_postprocessing(
        scores, cy_medfilt,
        filter_length_in_sec=filter_length_in_sec, time_decimals=time_decimals,
        lazy=lazy, storage_dir=storage_dir
    )


def apply_postprocessing(scores, postprocessing_fn, *, lazy=False, storage_dir=None, **kwargs):
    postprocessing_fn = partial(_postprocessing_fn_wrapper, postprocessing_fn=postprocessing_fn, **kwargs)
    if isinstance(scores, (str, Path)):
        scores = Path(scores)
        scores = lazy_sed_scores_loader(scores)
    elif lazy:
        scores = lazy_dataset.new(scores)
    if storage_dir is not None:
        storage_dir = Path(storage_dir)
        storage_dir.mkdir(exist_ok=True, parents=True)
    if lazy:
        scores = scores.map(postprocessing_fn)
        if storage_dir is not None:
            scores = lazy_dataset.zip(scores.keys(), scores).map(
                partial(_write_scores, storage_dir=storage_dir))
        return scores
    scores_processed = {}
    for audio_id in scores.keys():
        try:
            scores_processed[audio_id] = postprocessing_fn(scores[audio_id])
        except Exception as e:
            print(audio_id)
            raise e
        if storage_dir is not None:
            _write_scores((audio_id, scores_processed[audio_id]), storage_dir)
    return scores_processed


def _postprocessing_fn_wrapper(score_df, postprocessing_fn, **kwargs):
    timestamps, event_classes = validate_score_dataframe(score_df)
    score_arr = score_df[event_classes].to_numpy()
    score_arr_processed, timestamps_processed = postprocessing_fn(score_arr, timestamps, **kwargs)
    return create_score_dataframe(score_arr_processed, timestamps_processed, event_classes)


def _write_scores(audio_id_scores_df, storage_dir):
    audio_id, scores_df = audio_id_scores_df
    storage_path = Path(storage_dir) / f"{audio_id}.tsv"
    write_sed_scores(scores_df, storage_path)
    return scores_df
