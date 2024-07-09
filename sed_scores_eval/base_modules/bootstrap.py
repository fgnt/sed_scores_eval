import numpy as np
from sed_scores_eval.base_modules.io import parse_ground_truth, parse_audio_durations
from sed_scores_eval.utils import parallel


def bootstrap(
        eval_fn, scores=None, deltas=None, deltas_fn=None,
        n_bootstrap_samples=100, num_jobs=1,
        deltas_fn_kwargs=None, eval_fn_kwargs=None,
):
    if deltas_fn_kwargs is None:
        deltas_fn_kwargs = {}
    if eval_fn_kwargs is None:
        eval_fn_kwargs = {}
    if deltas is None:
        assert scores is not None
        assert deltas_fn is not None
        deltas = deltas_fn(
            scores=scores, num_jobs=num_jobs, **deltas_fn_kwargs,
        )
    return bootstrap_from_deltas(
        eval_fn, deltas,
        n_bootstrap_samples=n_bootstrap_samples, num_jobs=num_jobs,
        scores=None, **deltas_fn_kwargs, **eval_fn_kwargs,
    )


def bootstrap_from_deltas(
        eval_fn, deltas, *,
        n_bootstrap_samples=100, num_jobs=1,
        ground_truth=None, audio_durations=None,
        **eval_fn_kwargs,
):
    if isinstance(deltas, (list, tuple)):
        audio_ids = sorted(deltas[0].keys())
    else:
        audio_ids = sorted(deltas.keys())
    data_orig = {'deltas': deltas}
    data_samples = {'deltas': []}
    if ground_truth is not None:
        data_orig['ground_truth'] = parse_ground_truth(ground_truth)
        data_samples['ground_truth'] = []
    if audio_durations is not None:
        data_orig['audio_durations'] = parse_audio_durations(audio_durations)
        data_samples['audio_durations'] = []

    audio_ids_repeated = n_bootstrap_samples * audio_ids
    np.random.RandomState(0).shuffle(audio_ids_repeated)
    for i in range(n_bootstrap_samples):
        for key in data_samples:
            data_samples[key].append({})
        for j, audio_id in enumerate(audio_ids_repeated[i*len(audio_ids):(i+1)*len(audio_ids)]):
            for key in data_samples:
                if isinstance(data_orig[key], (list, tuple)):
                    if isinstance(data_samples[key][-1], dict):
                        data_samples[key][-1] = [{} for _ in range(len(data_orig[key]))]
                    for k in range(len(data_orig[key])):
                        data_samples[key][-1][k][f'{audio_id}_bootstrap{i}_clip{j}'] = data_orig[key][k][audio_id]
                else:
                    data_samples[key][-1][f'{audio_id}_bootstrap{i}_clip{j}'] = data_orig[key][audio_id]

    arg_keys = sorted(data_samples.keys())
    args = [data_samples[key] for key in arg_keys]
    ret = parallel.map(
        args, arg_keys=arg_keys,
        func=eval_fn, max_jobs=num_jobs,
        **eval_fn_kwargs,
    )
    if isinstance(ret[0], tuple):
        return list(zip(*ret))
    return ret


def _recursive_multiply(deltas, factor):
    if isinstance(deltas, dict):
        return {key: _recursive_multiply(deltas[key], factor) for key in deltas.keys()}
    return deltas * factor


def confidence_interval(bootstrapped_outputs, confidence=.9, axis=None):
    if isinstance(bootstrapped_outputs[0], dict):
        mean_low_high = {
            class_name: confidence_interval([
                output[class_name] for output in bootstrapped_outputs
            ])
            for class_name in bootstrapped_outputs[0]
        }
        return mean_low_high

    mean = float(np.mean(bootstrapped_outputs, axis=axis))
    low = float(np.percentile(
        bootstrapped_outputs, ((1 - confidence) / 2) * 100, axis=axis,
    ))
    high = float(np.percentile(
        bootstrapped_outputs, (confidence + ((1 - confidence) / 2)) * 100, axis=axis,
    ))
    return mean, low, high
