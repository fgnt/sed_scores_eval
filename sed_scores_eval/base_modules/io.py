from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import lazy_dataset
from sed_scores_eval.utils.scores import (
    get_unique_thresholds, extract_timestamps_and_classes_from_dataframe
)
from sed_scores_eval.base_modules.ground_truth import (
    onset_offset_times_to_indices
)


def parse_inputs(scores, ground_truth):
    assert isinstance(scores, (dict, str, Path, lazy_dataset.Dataset)), type(scores)
    if isinstance(scores, (str, Path)):
        scores = Path(scores)
        assert scores.is_dir(), scores
        scores = lazy_sed_scores_loader(scores)
    audio_ids = sorted(scores.keys())
    assert isinstance(ground_truth, (dict, str, Path)), type(ground_truth)
    if isinstance(ground_truth, (str, Path)):
        ground_truth = Path(ground_truth)
        assert ground_truth.is_file(), ground_truth
        ground_truth = read_ground_truth_events(ground_truth)
    assert sorted(ground_truth.keys()) == audio_ids, (
        set(audio_ids) - ground_truth.keys(), ground_truth.keys() - set(audio_ids))
    return scores, ground_truth, audio_ids


def write_sed_scores(scores, filepath, event_classes=None):
    """write sound event detection scores to tsv file

    Args:
        scores (pandas.DataFrame): containing onset and offset times
            of a score window in first two columns followed by sed score
            columns for each event class.
        filepath (str or pathlib.Path): path to file that is to be written
        event_classes (list of str): optional list of event classes used to
            assert correct event labels in scores DataFrame

    """
    score_data_frame_assertions(scores, event_classes=event_classes)
    scores.to_csv(filepath, sep='\t', index=False)


def score_data_frame_assertions(scores, event_classes=None):
    assert isinstance(scores, pd.DataFrame), type(scores)
    column_names = list(scores.columns)
    assert len(column_names) > 2, column_names
    assert column_names[0] == 'onset', column_names
    assert column_names[1] == 'offset', column_names
    if event_classes is not None:
        assert column_names[2:] == event_classes, (column_names, event_classes)
    onset_times = scores['onset'].to_numpy()
    offset_times = scores['offset'].to_numpy()
    assert (offset_times[:-1] == onset_times[1:]).all(), (onset_times, offset_times)


def read_sed_scores(filepath):
    scores = pd.read_csv(filepath, sep='\t')
    score_data_frame_assertions(scores)
    return scores


def lazy_sed_scores_loader(dir_path):
    """loader for sound event detection files in a directory

    Args:
        dir_path (str or pathlib.Path): path to directory with sound event
            detection files
    """
    dir_path = Path(dir_path)
    score_files = {}
    for file in sorted(dir_path.iterdir()):
        assert file.is_file(), file
        assert file.name.endswith('.tsv'), file
        score_files[file.name[:-len('.tsv')]] = str(file)
    scores = lazy_dataset.new(score_files)
    return scores.map(read_sed_scores)


def read_ground_truth_events(filepath):
    """read ground truth events from tsv file

    Args:
        filepath (str or pathlib.Path): path to file that is to be read.

    Returns:
        ground_truth (dict of lists of tuples): list of ground truth event
            tuples (onset, offset, event class) for each audio clip.

    """
    ground_truth = {}
    file = pd.read_csv(filepath, sep='\t')
    assert [
        name in list(file.columns)
        for name in ['filename', 'onset', 'offset', 'event_label']
    ], list(file.columns)
    for filename, onset, offset, event_label in zip(
        file['filename'], file['onset'], file['offset'], file['event_label']
    ):
        example_id = filename.rsplit('.', maxsplit=1)[0]
        if example_id not in ground_truth:
            ground_truth[example_id] = []
        if isinstance(event_label, str):
            assert len(event_label) > 0
            ground_truth[example_id].append([
                float(onset), float(offset), event_label
            ])
        else:
            # file without active events
            assert np.isnan(event_label), event_label
    return ground_truth


def read_ground_truth_tags(filepath):
    """read ground truth tags from tsv file

    Args:
        filepath (str or pathlib.Path): path to file that is to be read.

    Returns:
        tags:
        class_counts:

    """
    tags = {}
    file = pd.read_csv(filepath, sep='\t')
    assert [
        name in list(file.columns)
        for name in ['filename', 'event_label']
    ], list(file.columns)
    class_counts = {}
    for filename, event_labels in zip(file['filename'], file['event_label']):
        example_id = filename.rsplit('.', maxsplit=1)[0]
        if example_id not in tags:
            tags[example_id] = []
        if len(event_labels) > 0:
            event_labels = event_labels.split(',')
            for label in event_labels:
                tags[example_id].append(label)
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1
    return tags, class_counts


def read_audio_durations(filepath):
    """read audio clip durations from tsv file

    Args:
        filepath (str or pathlib.Path): path to file that is to be read.

    Returns:
        audio_duration (dict of floats): audio duration in seconds for each
            audio file

    """
    audio_duration = {}
    file = pd.read_csv(filepath, sep='\t')
    assert [
        name in list(file.columns) for name in ['filename', 'duration']
    ], list(file.columns)
    for filename, duration in zip(file['filename'], file['duration']):
        example_id = filename.rsplit('.', maxsplit=1)[0]
        audio_duration[example_id] = float(duration)
    return audio_duration


def write_detection(
        scores, thresholds, filepath, audio_format='wav'
):
    """perform sound event detection and write detected events to tsv file

    Args:
        scores (dict of pandas.DataFrame): each DataFrame containing onset and
            offset times of a score window in first two columns followed by
            sed score columns for each event class. Dict keys have to be
            filenames without audio format ending.
        thresholds (np.array): decision thresholds for each event class.
        filepath (str or pathlib.Path): path to file that is to be written/extended.
        audio_format: the audio format that is required to reconstruct the
            filename from dict keys.

    """
    assert hasattr(scores, 'keys'), 'scores must implement scores.keys()'
    assert callable(scores.keys), 'scores must implement scores.keys()'
    keys = sorted(scores.keys())
    _, event_classes = extract_timestamps_and_classes_from_dataframe(
        scores[keys[0]])
    if isinstance(thresholds, dict):
        thresholds = [thresholds[event_class] for event_class in event_classes]
    if not np.isscalar(thresholds):
        assert isinstance(thresholds, (list, np.ndarray)), thresholds
        thresholds = np.array(thresholds)
        assert thresholds.shape == (len(event_classes),), thresholds.shape
    filepath = Path(filepath)
    if not filepath.exists() or filepath.stat().st_size == 0:
        with Path(filepath).open('w') as fid:
            fid.write('filename\tonset\toffset\tevent_label\n')

    with filepath.open('a') as fid:
        for key in keys:
            scores_i = scores[key]
            score_data_frame_assertions(scores_i, event_classes=event_classes)
            if event_classes is None:
                event_classes = list(scores_i.columns)[2:]
                assert thresholds.shape == (len(event_classes),)
            onset_times = scores_i['onset'].to_numpy()
            offset_times = scores_i['offset'].to_numpy()
            scores_i = scores_i[event_classes].to_numpy()
            detections = scores_i > thresholds
            zeros = np.zeros_like(detections[:1, :])
            detections = np.concatenate((zeros, detections, zeros), axis=0).astype(np.float)
            change_points = detections[1:] - detections[:-1]
            event_list = []
            for k in np.argwhere(np.abs(change_points).max(0) > .5).flatten():
                onsets = np.argwhere(change_points[:, k] > .5).flatten()
                offsets = np.argwhere(change_points[:, k] < -.5).flatten()
                assert len(onsets) == len(offsets) > 0
                for onset, offset in zip(onsets, offsets):
                    event_list.append((
                        onset_times[onset], offset_times[offset-1],
                        event_classes[k]
                    ))
            event_list = sorted(event_list)
            for t_on, t_off, event_label in event_list:
                fid.write(f'{key}.{audio_format}\t{t_on}\t{t_off}\t{event_label}\n')


def write_detections_for_multiple_thresholds(
        scores, thresholds, dir_path, audio_format='wav', score_transform=None,
):
    """writes a detection for multiple thresholds (operating points) as
    required by the psds_eval package (https://github.com/audioanalytic/psds_eval).
    This function is primarily used for testing purposes.

    Args:
        scores (dict of pandas.DataFrame): each DataFrame containing onset and
            offset times of a score window in first two columns followed by
            sed score columns for each event class. Dict keys have to be
            filenames without audio format ending.
        thresholds (np.array): an array of decision thresholds for each of
            which a detection file is written.
        dir_path (str or pathlib.Path): path to directory where to save
            detection files.
        audio_format: the audio format that is required to reconstruct the
            filename from dict keys.
        score_transform:

    """
    assert hasattr(scores, 'keys'), 'scores must implement scores.keys()'
    assert callable(scores.keys), 'scores must implement scores.keys()'
    keys = sorted(scores.keys())
    thresholds = np.array(thresholds)
    assert thresholds.ndim == 1, thresholds.shape
    dir_path = Path(dir_path)

    if score_transform is not None:
        if isinstance(score_transform, (str, Path)):
            score_transform = read_score_transform(score_transform)
        assert callable(score_transform), score_transform
        if isinstance(scores, lazy_dataset.Dataset):
            scores = scores.map(score_transform)
        else:
            scores = {
                key: score_transform(scores_i)
                for key, scores_i in scores.items()
            }
    for key in keys:
        scores_i = scores[key]
        for threshold in thresholds:
            write_detection(
                {key: scores_i}, threshold,
                dir_path / '{:.3f}.tsv'.format(threshold),
                audio_format=audio_format,
            )


def write_score_transform(
        scores, ground_truth, filepath,
        num_breakpoints=1001, min_score=0., max_score=1.
):
    """compute and save a piecewise-linear score transform which is supposed
    to uniformly distribute scores from within ground truth events between 0
    and 1. This allows to obtain smoother PSD-ROC curves with linearly spaced
    thresholds.

    Args:
        scores (dict of pandas.DataFrames): score DataFrames for each audio
            clip of a data set. Each DataFrame contains onset and offset times
            of a score window  in first two columns followed by sed score
            columns for each event class.
        ground_truth (dict of lists of tuples): list of ground truth event
            tuples (onset, offset, event class) for each audio clip.
        filepath (str or pathlib.Path): path to file that is to be written.
        num_breakpoints: the number of breakpoints in the piecewise-linear
            transformation function.
        min_score: the first value (where y=x) in the transformation.
        max_score: the last value (where y=x) in the transformation.

    """
    scores, ground_truth, keys = parse_inputs(scores, ground_truth)
    scores_at_positives = {}
    for key in keys:
        scores_for_key = scores[key]
        score_data_frame_assertions(scores_for_key)
        onset_times = scores_for_key['onset'].to_numpy()
        offset_times = scores_for_key['offset'].to_numpy()
        timestamps = np.concatenate((onset_times, offset_times[-1:]))
        for (t_on, t_off, event_label) in ground_truth[key]:
            idx_on, idx_off = onset_offset_times_to_indices(
                onset_time=t_on, offset_time=t_off, timestamps=timestamps
            )
            if event_label not in scores_at_positives:
                scores_at_positives[event_label] = []
            scores_at_positives[event_label].append(
                scores_for_key[event_label].to_numpy()[idx_on:idx_off])
    output_scores = np.linspace(min_score, max_score, num_breakpoints)
    score_transform = [output_scores]
    event_classes = sorted(scores_at_positives.keys())
    for event_class in event_classes:
        scores_k = np.concatenate(scores_at_positives[event_class])
        thresholds, *_ = get_unique_thresholds(scores_k)
        assert len(thresholds) >= num_breakpoints, (len(thresholds), num_breakpoints)
        breakpoint_indices = np.linspace(
            0, len(thresholds), num_breakpoints)[1:-1].astype(np.int)
        assert (thresholds[breakpoint_indices] >= min_score).all(), (
            np.min(thresholds[breakpoint_indices]), min_score)
        assert (thresholds[breakpoint_indices] <= max_score).all(), (
            np.max(thresholds[breakpoint_indices]), max_score)
        breakpoints = np.concatenate((
            [min_score], thresholds[breakpoint_indices], [max_score]
        ))
        score_transform.append(breakpoints)
    score_transform = pd.DataFrame(
        np.array(score_transform).T, columns=['y', *event_classes])
    score_transform.to_csv(filepath, sep='\t', index=False)


def read_score_transform(filepath):
    """read a piecewise linear score transform from tsv file

    Args:
        filepath: path to tsv file as written by write_score_transform

    Returns:
        score_transform: function which takes scores as pd.DataFrame and
            returns the transformed scores as pd.DataFrame

    """
    transform = pd.read_csv(filepath, sep='\t')
    column_names = list(transform.columns)
    assert len(column_names) > 1, column_names
    assert column_names[0] == 'y', column_names
    event_classes = column_names[1:]
    y = transform['y'].to_numpy()

    def score_transform(scores):
        score_data_frame_assertions(scores, event_classes=event_classes)
        transformed_scores = [
            scores['onset'].to_numpy(), scores['offset'].to_numpy()
        ]
        for event_class in event_classes:
            x = transform[event_class].to_numpy()
            transformed_scores.append(interp1d(
                x, y, kind='linear',
            )(scores[event_class]))
        transformed_scores = pd.DataFrame(
            np.array(transformed_scores).T,
            columns=['onset', 'offset', *event_classes],
        )
        return transformed_scores

    return score_transform
