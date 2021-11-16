import numpy as np
import pandas as pd
from pathlib import Path
from sed_scores_eval.base_modules.io import (
    parse_inputs, write_detections_for_multiple_thresholds
)
from sed_scores_eval.utils.scores import extract_timestamps_and_classes_from_dataframe


def approximate_psds(
        scores, ground_truth, audio_durations,
        thresholds=np.linspace(.01, .99, 50), *,
        dtc_threshold, gtc_threshold, cttc_threshold=None,
        alpha_ct=.0, alpha_st=.0, unit_of_time='hour', max_efpr=100.,
        audio_format='wav', score_transform=None,
):
    import tempfile
    from psds_eval import PSDSEval
    assert isinstance(ground_truth, (str, Path)), type(ground_truth)
    assert isinstance(audio_durations, (str, Path)), type(audio_durations)
    scores, _, keys = parse_inputs(scores, ground_truth)
    ground_truth = pd.read_csv(ground_truth, sep="\t")
    audio_durations = pd.read_csv(audio_durations, sep="\t")
    _, event_classes = extract_timestamps_and_classes_from_dataframe(
        scores[keys[0]])
    psds_eval = PSDSEval(
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=1. if cttc_threshold is None else cttc_threshold,
        ground_truth=ground_truth,
        metadata=audio_durations,
        duration_unit=unit_of_time,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_path = Path(tmp_dir)
        write_detections_for_multiple_thresholds(
            scores, thresholds, dir_path, audio_format=audio_format,
            score_transform=score_transform,
        )

        psds_eval.clear_all_operating_points()

        for i, tsv in enumerate(dir_path.glob('*.tsv')):
            print(f"Adding operating point {i+1}/{len(thresholds)}", end="\r")
            threshold = float(tsv.name[:-len('.tsv')])
            det = pd.read_csv(tsv, sep="\t")
            info = {"name": f"Op {i+1:02d}", "threshold": threshold}
            psds_eval.add_operating_point(det, info=info)

        # compute the PSDS of the system represented by its operating points
        psds_ = psds_eval.psds(
            max_efpr=max_efpr,
            alpha_st=alpha_st,
            alpha_ct=alpha_ct,
        )
        efpr = psds_.plt.xp
        etpr = psds_.plt.yp
        _, _, tpr_vs_efpr = psds_eval.psd_roc_curves(alpha_ct=alpha_ct)

        single_class_psd_rocs = {
            event_classes[i]: (tpr_vs_efpr.yp[i], tpr_vs_efpr.xp)
            for i in range(len(tpr_vs_efpr.yp))
        }
    return psds_.value, (etpr, efpr), single_class_psd_rocs
