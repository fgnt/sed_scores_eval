from pathlib import Path
from sed_scores_eval.base_modules.io import parse_inputs, write_detection


def fscore(
        scores, ground_truth, threshold, *, collar, offset_collar_rate=0.,
        audio_format='wav',
):
    import tempfile
    import sed_eval
    import dcase_util

    assert isinstance(ground_truth, (str, Path)), type(ground_truth)
    ground_truth = str(ground_truth)
    scores, *_ = parse_inputs(scores, ground_truth)
    tmp_fid = tempfile.NamedTemporaryFile(delete=False)
    tmp_filepath = Path(tmp_fid.name)
    tmp_fid.close()
    try:
        write_detection(
            scores=scores, thresholds=threshold,
            filepath=tmp_filepath, audio_format=audio_format,
        )
        reference_event_list = sed_eval.io.load_event_list(
            filename=ground_truth
        )
        reference_event_list = dcase_util.containers.MetaDataContainer(
            [entry for entry in reference_event_list if
             entry['event_label'] is not None]
        )
        estimated_event_list = sed_eval.io.load_event_list(
            filename=str(tmp_filepath)
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
        tmp_filepath.unlink(missing_ok=True)
