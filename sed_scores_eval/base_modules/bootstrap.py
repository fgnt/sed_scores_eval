import numpy as np
from sed_scores_eval.utils import parallel


def bootstrap_from_deltas(
        metric_fn, deltas, *,
        n_folds=5, n_iterations=20, num_jobs=1,
        **metric_fn_kwargs,
):
    if isinstance(deltas, (list, tuple)):
        audio_ids = sorted(deltas[0].keys())
    else:
        audio_ids = sorted(deltas.keys())
    split_indices = np.linspace(0, len(audio_ids), n_folds+1).astype(int)
    audio_id_fractions = []
    for i in range(n_iterations):
        np.random.RandomState(i).shuffle(audio_ids)
        for j in range(n_folds):
            audio_id_fractions.append(list(
                audio_ids[:split_indices[j]] + audio_ids[split_indices[j+1]:]
            ))

    if isinstance(deltas, (list, tuple)):
        deltas_fractions = [
            [
                {audio_id: delts[audio_id] for audio_id in audio_id_fraction}
                for delts in deltas
            ]
            for audio_id_fraction in audio_id_fractions
        ]
    else:
        deltas_fractions = [
            {audio_id: deltas[audio_id] for audio_id in audio_id_fraction}
            for audio_id_fraction in audio_id_fractions
        ]
    return list(zip(*parallel.map(
        deltas_fractions, arg_keys='deltas',
        func=metric_fn, max_jobs=num_jobs,
        **metric_fn_kwargs,
    )))


def confidence_interval(bootstrapped_outputs, confidence=.9):
    if isinstance(bootstrapped_outputs[0], dict):
        mean_low_high = {
            class_name: confidence_interval([
                output[class_name] for output in bootstrapped_outputs
            ])
            for class_name in bootstrapped_outputs[0]
        }
        return mean_low_high

    mean = np.mean(bootstrapped_outputs)
    low = np.percentile(
        bootstrapped_outputs, ((1 - confidence) / 2) * 100
    )
    high = np.percentile(
        bootstrapped_outputs, (confidence + ((1 - confidence) / 2)) * 100
    )
    return float(mean), float(low), float(high)

