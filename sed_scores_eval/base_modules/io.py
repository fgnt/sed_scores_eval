from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from collections import defaultdict
import lazy_dataset
from urllib.request import urlretrieve
from sed_scores_eval.base_modules.scores import (
    create_score_dataframe, validate_score_dataframe,
)
from sed_scores_eval.base_modules.detection import scores_to_event_list
from sed_scores_eval.base_modules.ground_truth import onset_offset_times_to_indices


def parse_inputs(scores, ground_truth, *, tagging=False):
    """read scores and ground_truth from files if string or path provided and
    validate audio ids

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.

    Returns:
        scores:
        ground_truth:
        audio_ids:

    """
    scores, audio_ids = parse_scores(scores)
    ground_truth = parse_ground_truth(ground_truth, tagging=tagging, audio_ids=audio_ids)
    return scores, ground_truth, audio_ids


def parse_scores(scores):
    if not isinstance(scores, (dict, str, Path, lazy_dataset.Dataset)):
        raise ValueError(
            f'scores must be dict, str, pathlib.Path or lazy_dataset.Dataset '
            f'but {type(scores)} was given.'
        )
    if isinstance(scores, (str, Path)):
        scores = Path(scores)
        scores = lazy_sed_scores_loader(scores)
    audio_ids = sorted(scores.keys())
    return scores, audio_ids


def parse_ground_truth(
        ground_truth, *,
        tagging=False, audio_ids=None, additional_ids_ok=False
):
    if not isinstance(ground_truth, (dict, str, Path)):
        raise ValueError(
            f'ground_truth must be dict, str or Path but {type(ground_truth)} '
            f'was given.'
        )
    if isinstance(ground_truth, (str, Path)):
        ground_truth = Path(ground_truth)
        if tagging:
            ground_truth, _ = read_ground_truth_tags(ground_truth)
        else:
            ground_truth = read_ground_truth_events(ground_truth)
    if not tagging:
        assert_non_connected_events(ground_truth)
    if audio_ids is not None:
        if additional_ids_ok:
            ground_truth = {key: ground_truth[key] for key in audio_ids}
        elif not (ground_truth.keys() == set(audio_ids)):
            raise ValueError(
                f'ground_truth audio ids do not match audio_ids. '
                f'Missing ids: {set(audio_ids) - ground_truth.keys()}. '
                f'Additional ids: {ground_truth.keys() - set(audio_ids)}.'
            )
    return ground_truth


def assert_non_connected_events(ground_truth):
    for clip_id, events in ground_truth.items():
        per_class_events = defaultdict(list)
        for event in events:
            per_class_events[event[2]].append(event)
        for event_class, class_events in per_class_events.items():
            class_events = sorted(class_events)
            current_offset = -1e6
            for event in class_events:
                assert event[0] > current_offset, f'Connected/overlapping {event[2]} events found for clip {clip_id}: {class_events}'
                current_offset = event[1]


def parse_audio_durations(audio_durations, *, audio_ids=None, additional_ids_ok=False):
    if not isinstance(audio_durations, (dict, str, Path)):
        raise ValueError(
            f'audio_durations must be dict, str or Path but '
            f'{type(audio_durations)} was given.'
        )
    if isinstance(audio_durations, (str, Path)):
        audio_durations = Path(audio_durations)
        assert audio_durations.is_file(), audio_durations
        audio_durations = read_audio_durations(audio_durations)
    if audio_ids is not None:
        if additional_ids_ok:
            audio_durations = {key: audio_durations[key] for key in audio_ids}
        elif not (audio_durations.keys() == set(audio_ids)):
            raise ValueError(
                f'audio_durations audio ids do not match audio_ids. '
                f'Missing ids: {set(audio_ids) - audio_durations.keys()}. '
                f'Additional ids: {audio_durations.keys() - set(audio_ids)}.'
            )
    return audio_durations


def write_sed_scores(scores, storage_path, *, timestamps=None, event_classes=None):
    """write sound event detection scores to tsv file

    Args:
        scores ((dict of) pandas.DataFrame): containing onset and offset times
            of a score window in first two columns followed by sed score
            columns for each event class. If dict keys are expected to be
            audio ids with corresponding data frames as values.
        storage_path (str or pathlib.Path): path to directory/file that is to be written
        timestamps (np.ndarray or list of float): optional list of timestamps
            to be compared with timestamps in scores DataFrame
        event_classes (list of str): optional list of event classes used to
            assert correct event labels in scores DataFrame

    """
    if isinstance(scores, dict):
        storage_path = Path(storage_path)
        storage_path.mkdir(exist_ok=True, parents=True)
        for audio_id, c_scores in scores.items():
            write_sed_scores(
                c_scores, storage_path / (audio_id + '.tsv'),
                timestamps=timestamps, event_classes=event_classes
            )
        return
    if not isinstance(scores, (np.ndarray, pd.DataFrame)):
        raise ValueError(
            f'scores must be np.ndarray or pd.DataFrame but {type(scores)}'
            f'was given.'
        )
    if isinstance(scores, np.ndarray):
        if timestamps is None:
            raise ValueError(
                f'timestamps must not be None if scores is np.ndarray'
            )
        if event_classes is None:
            raise ValueError(
                f'event_classes must not be None if scores is np.ndarray'
            )
        scores = create_score_dataframe(scores, timestamps, event_classes)
    validate_score_dataframe(scores, timestamps=timestamps, event_classes=event_classes)
    scores.to_csv(storage_path, sep='\t', index=False)


def read_sed_scores(filepath):
    """read scores from file(s)

    Args:
        filepath: path to tsv-file of a single utterance or to dir with tsv-files for the whole eval set

    Returns:

    """
    filepath = Path(filepath)
    if filepath.is_dir():
        scores = {}
        for file in sorted(filepath.iterdir()):
            if not file.is_file() or not file.name.endswith('.tsv'):
                raise ValueError(f'dir_path must only contain tsv files but contains {file}')
            scores[file.name[:-len('.tsv')]] = read_sed_scores(file)
        return scores
    scores = pd.read_csv(filepath, sep='\t')
    validate_score_dataframe(scores)
    return scores


def lazy_sed_scores_loader(dir_path):
    """lazy loader for sound event detection files in a directory. This is
    particularly useful if scores do not fit in memory for all audio files
    simultaneously.

    Args:
        dir_path (str or pathlib.Path): path to directory with sound event
            detection files
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(str(dir_path))
    score_files = {}
    for file in sorted(dir_path.iterdir()):
        if not file.is_file() or not file.name.endswith('.tsv'):
            raise ValueError(f'dir_path must only contain tsv files but contains {file}')
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
    if isinstance(filepath, pd.DataFrame):
        file = filepath
    else:
        file = pd.read_csv(filepath, sep='\t')
    if not all([
        name in list(file.columns)
        for name in ['filename', 'onset', 'offset', 'event_label']
    ]):
        raise ValueError(
            f'ground_truth events file must contain columns "filename", '
            f'"onset", "offset" and "event_label" but only columns '
            f'{list(file.columns)} were found.'
        )
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
        tags (dict of lists): list of active events for each audio file.
        class_counts (dict of ints): number of files in which event_class is
            active for each event_class

    """
    tags = {}
    file = pd.read_csv(filepath, sep='\t')
    if 'filename' not in file.columns or (
            'event_label' not in file.columns
            and 'event_labels' not in file.columns
    ):
        raise ValueError(
            f'ground_truth tags file must contain columns "filename", '
            f'and "event_label" or "event_labels" but only columns '
            f'{list(file.columns)} were found.'
        )
    event_labels_key = "event_labels" if "event_labels" in file.columns else "event_label"
    class_counts = {}
    for filename, event_labels in zip(file['filename'], file[event_labels_key]):
        example_id = filename.rsplit('.', maxsplit=1)[0]
        if example_id not in tags:
            tags[example_id] = []
        if isinstance(event_labels, str):
            if event_labels_key == "event_labels":
                event_labels = event_labels.split(',')
            else:
                event_labels = [event_labels]
            for label in event_labels:
                tags[example_id].append(label)
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1
        else:
            # file without active events
            assert np.isnan(event_labels), event_labels
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
        scores, threshold, filepath, audio_format='wav',
):
    """perform thresholding of sound event detection scores and write detected
    events to tsv file

    Args:
        scores (dict of pandas.DataFrame): each DataFrame containing onset and
            offset times of a score window in first two columns followed by
            sed score columns for each event class. Dict keys have to be
            filenames without audio format ending.
        threshold ((dict of) float): threshold that is to be evaluated.
        filepath (str or pathlib.Path): path to file that is to be written/extended.
        audio_format: the audio format that is required to reconstruct the
            filename from audio ids/keys.

    """
    if not hasattr(scores, 'keys') or not callable(scores.keys):
        raise ValueError('scores must implement scores.keys()')
    keys = sorted(scores.keys())
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
    filepath = Path(filepath)
    if not filepath.exists() or filepath.stat().st_size == 0:
        with Path(filepath).open('w') as fid:
            fid.write('filename\tonset\toffset\tevent_label\n')

    with filepath.open('a') as fid:
        event_lists = scores_to_event_list(scores, thresholds=threshold)
        for key, event_list in event_lists.items():
            for t_on, t_off, event_label in event_list:
                fid.write(
                    f'{key}.{audio_format}\t{t_on}\t{t_off}\t{event_label}\n')


def write_detections_for_multiple_thresholds(
        scores, thresholds, dir_path, audio_format='wav', score_transform=None,
        threshold_decimals=3,
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
            filename from audio ids/keys.
        score_transform:

    """
    if not hasattr(scores, 'keys') or not callable(scores.keys):
        raise ValueError('scores must implement scores.keys()')
    keys = sorted(scores.keys())
    thresholds = np.asanyarray(thresholds)
    if thresholds.ndim != 1:
        raise ValueError(
            f'thresholds must be a 1-dimensional array but has shape '
            f'{thresholds.shape}.'
        )
    assert np.all(np.abs(thresholds - np.round(thresholds, threshold_decimals)) < 1e-15), (threshold_decimals, thresholds)
    assert np.all(thresholds == np.unique(thresholds)), thresholds
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    if score_transform is not None:
        if isinstance(score_transform, (str, Path)):
            score_transform = read_score_transform(score_transform)
        if not callable(score_transform):
            raise ValueError('score_transform must be callable.')
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
                dir_path / '{:.Xf}.tsv'.replace('X', str(threshold_decimals)).format(threshold),
                audio_format=audio_format,
            )


def write_score_transform(
        scores, ground_truth, filepath,
        num_breakpoints=10, min_score=0., max_score=1.,
        classwise_transform=False,
):
    """compute and save a piecewise-linear score transform which is supposed
    to uniformly distribute scores from within ground truth events between 0
    and 1. This allows to obtain smoother PSD-ROC curve approximations when
    using the psds_eval package (https://github.com/audioanalytic/psds_eval)
    with linearly spaced thresholds.
    This function is primarily used for testing purposes.

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
        classwise_transform: If True, use separate transformations for scores
            from different event classes

    """
    scores, ground_truth, keys = parse_inputs(scores, ground_truth)
    scores_dict = {}
    event_classes = None
    for key in keys:
        scores_for_key = scores[key]
        timestamps, event_classes = validate_score_dataframe(
            scores_for_key, event_classes=event_classes)
        for (t_on, t_off, event_label) in ground_truth[key]:
            if event_label not in scores_dict:
                scores_dict[event_label] = []
            idx_on, idx_off = onset_offset_times_to_indices(
                onset_time=t_on, offset_time=t_off, timestamps=timestamps
            )
            scores_dict[event_label].append(scores_for_key[event_label][idx_on:idx_off])
    scores_dict = {
        event_class: np.concatenate(scores_dict[event_class])
        for event_class in event_classes
    }
    step = (max_score-min_score)/num_breakpoints
    output_scores = np.concatenate((
        [min_score],
        np.linspace(min_score, max_score, num_breakpoints, endpoint=False) + step/2,
        [max_score],
    ))
    output_scores = np.round(output_scores, decimals=12)
    score_transform = [output_scores]

    def get_breakpoints(score_arr):
        score_arr = np.sort(score_arr[score_arr > .0])
        breakpoints = score_arr[np.linspace(0, len(score_arr), num_breakpoints+1, dtype=int)[:-1]]
        return np.concatenate((
            [min_score], breakpoints, [max_score]
        ))

    if classwise_transform:
        for event_class in event_classes:
            score_transform.append(get_breakpoints(
                scores_dict[event_class]
            ))
    else:
        score_arr = np.concatenate([
            scores_dict[event_class] for event_class in event_classes])
        score_transform.extend(
            len(event_classes) * [get_breakpoints(score_arr)]
        )
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
        validate_score_dataframe(scores, event_classes=event_classes)
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


def download_test_data():
    from sed_scores_eval import package_dir
    import zipfile
    tests_dir_path = package_dir / 'tests'
    if (tests_dir_path / 'data').exists():
        print('Test data already exists.')
        return
    print('Download test data')
    zip_file_path = tests_dir_path / 'data.zip'
    urlretrieve(
        'https://go.upb.de/sed_scores_eval_test_data',
        filename=str(zip_file_path)
    )
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(tests_dir_path)
    zip_file_path.unlink()
    print('Download successful')
