import pandas as pd
from pathlib import Path
from sed_scores_eval.base_modules.io import parse_inputs, write_detection


def metrics(
        scores, ground_truth, threshold, *, collar, offset_collar_rate=0.,
):
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
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        threshold ((dict of) float): threshold that is to be evaluated.
        collar (float): allowed onset and (at least) allowed offset deviation
            in seconds
        offset_collar_rate (float): (at least) allowed offset deviation as a
            ratio of the length of the ground truth event, with the actual
            allowed offset deviation being:
            offset_collar_for_gt_event = max(
                collar, offset_collar_rate*length_of_gt_event_in_seconds
            )

    Returns (sed_eval.sound_event.EventBasedMetrics): collar-based metrics

    """
    import tempfile
    import sed_eval
    import dcase_util

    assert isinstance(ground_truth, (str, Path)), type(ground_truth)
    ground_truth = str(ground_truth)
    audio_format = pd.read_csv(ground_truth, sep="\t")['filename'][0].split('.')[-1]
    scores, *_ = parse_inputs(scores, ground_truth)
    tmp_fid = tempfile.NamedTemporaryFile(delete=False)
    tmp_filepath = Path(tmp_fid.name)
    tmp_fid.close()
    try:
        write_detection(
            scores=scores, threshold=threshold,
            filepath=tmp_filepath, audio_format=audio_format,
        )
        reference_event_list = sed_eval.io.load_event_list(
            filename=ground_truth, file_format='CSV',
        )
        reference_event_list = dcase_util.containers.MetaDataContainer(
            [entry for entry in reference_event_list if
             entry['event_label'] is not None]
        )
        estimated_event_list = sed_eval.io.load_event_list(
            filename=str(tmp_filepath), file_format='CSV'
        )

        all_data = dcase_util.containers.MetaDataContainer()
        all_data += reference_event_list
        all_data += estimated_event_list

        event_labels = all_data.unique_event_labels

        # Start evaluating
        # Create metrics classes, define parameters
        event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=event_labels,
            t_collar=collar, percentage_of_length=offset_collar_rate,
            event_matching_type='optimal',
        )

        # Go through files
        for filename in all_data.unique_files:
            reference_event_list_for_current_file = reference_event_list.filter(
                filename=filename
            )

            estimated_event_list_for_current_file = estimated_event_list.filter(
                filename=filename
            )
            event_based_metrics.evaluate(
                reference_event_list=reference_event_list_for_current_file,
                estimated_event_list=estimated_event_list_for_current_file
            )
        return event_based_metrics
    finally:
        if tmp_filepath.exists():
            tmp_filepath.unlink()
