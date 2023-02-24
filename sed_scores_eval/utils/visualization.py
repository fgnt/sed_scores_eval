import matplotlib.pyplot as plt


def plot_psd_roc(
        psd_roc,
        *,
        dtc_threshold, gtc_threshold, cttc_threshold,
        alpha_ct, alpha_st, unit_of_time, max_efpr, psds,
        axes=None, filename=None,
        **kwargs
):
    """Shows (or saves) the PSD-ROC with optional standard deviation.

    This function is an adjustment from psds_eval.plot_psd_roc!

    When the plot is generated the area under PSD-ROC is highlighted.
    The plot is affected by the values used to compute the metric:
    max_efpr, alpha_ST and alpha_CT

    Args:
        psd_roc (tuple): The psd_roc that is to be plotted
        alpha_ct:
        alpha_st:
        unit_of_time:
        max_efpr:
        psds:
        axes (matplotlib.axes.Axes): matplotlib axes used for the plot
        filename (str): if provided a file will be saved with this name
        kwargs (dict): can set figsize
    """

    if not isinstance(psd_roc, tuple) or len(psd_roc) != 2:
        raise ValueError("The psd roc needs to be given as a tuple (etpr,efpr)")
    if axes is not None and not isinstance(axes, plt.Axes):
        raise ValueError("The give axes is not a matplotlib.axes.Axes")

    show = False
    if axes is None:
        fig = plt.figure(figsize=kwargs.get("figsize", (7, 7)))
        axes = fig.add_subplot()
        show = True

    axes.vlines(max_efpr, ymin=0, ymax=1.0, linestyles='dashed')
    etpr, efpr = psd_roc
    axes.step(efpr, etpr, 'b-', label='PSD-ROC', where="post")
    axes.fill_between(
        efpr, y1=etpr, y2=0, label="AUC", alpha=0.3, color="tab:blue",
        linewidth=3, step="post",
    )
    axes.set_xlim([0, max_efpr])
    axes.set_ylim([0, 1.0])
    axes.legend()
    axes.set_ylabel("eTPR")
    axes.set_xlabel(f"eFPR per {unit_of_time}")
    if cttc_threshold is None:
        cttc_threshold = "na"
    else:
        cttc_threshold = f"{cttc_threshold:.2f}"
    axes.set_title(
        f"PSDS: {psds:.5f}\n"
        f"dtc: {dtc_threshold:.2f}, "
        f"gtc: {gtc_threshold:.2f}, "
        f"cttc: {cttc_threshold}, "
        f"alpha_st: {alpha_st:.2f}, "
        f"alpha_ct: {alpha_ct:.2f}, "
        f"max_efpr: {max_efpr}"
    )
    axes.grid()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close()
