"""
Make figures for manuscript.
"""
import pandas as pd
import matplotlib
from pylab import *
from conf_analysis.behavior import metadata
from conf_analysis.meg import decoding_plots as dp
from joblib import Memory
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_rel, linregress

memory = Memory(location=metadata.cachedir, verbose=-1)

def_ig = slice(0.4, 1.1)


def plotwerr(pivottable, *args, ls="-", label=None, **kwargs):
    N = pivottable.shape[0]
    x = pivottable.columns.values
    mean = pivottable.mean(0).values
    std = pivottable.std(0).values
    sem = std / (N ** 0.5)
    plot(x, mean, *args, label=label, **kwargs)
    if "alpha" in kwargs:
        del kwargs["alpha"]
    if "color" in kwargs:
        color = kwargs["color"]
        del kwargs["color"]
        fill_between(
            x,
            mean + sem,
            mean - sem,
            facecolor=color,
            edgecolor="none",
            alpha=0.5,
            **kwargs
        )
    else:
        fill_between(x, mean + sem, mean - sem, edgecolor="none", alpha=0.5, **kwargs)
    # for col in pivottable:
    #    sem = pivottable.loc[:, col].std() / pivottable.shape[0] ** 0.5
    #    m = pivottable.loc[:, col].mean()
    #    plot([col, col], [m - sem, m + sem], *args, **kwargs)


def draw_sig(
    ax, pivottable, y=0, color="k", fdr=False, lw=2, conjunction=None, **kwargs
):
    from scipy.stats import ttest_1samp

    p_sig = ttest_1samp(pivottable, 0)[1]
    if fdr:
        from mne.stats import fdr_correction

        id_sig, _ = fdr_correction(p_sig)
        id_sig = list(id_sig)
    else:
        id_sig = list(p_sig < 0.05)

    if conjunction is not None:
        p_con_sig = ttest_1samp(conjunction, 0)[1]
        id_con_sig = p_con_sig < 0.05
        id_sig = list(np.array(id_sig) & id_con_sig)
    x = pivottable.columns.values
    d = np.where(np.diff([False] + id_sig + [False]))[0]
    dx = np.diff(x).astype(float)[0] / 10
    # xb = np.linspace(x.min()-dx, x.max()+dx, 5000)
    for low, high in zip(d[0:-1:2], d[1::2]):

        ax.plot([x[low] - dx, x[high - 1] + dx], [y, y], color=color, lw=lw, **kwargs)
    return p_sig


def _stream_palette():
    rois = [
        "vfcPrimary",
        "vfcEarly",
        "vfcV3ab",
        "vfcIPS01",
        "vfcIPS23",
        "JWG_aIPS",
        "vfcLO",
        "vfcTO",
        "vfcVO",
        "vfcPHC",
        "JWG_IPS_PCeS",
        "JWG_M1",
    ]
    return {
        roi: color
        for roi, color in zip(
            rois, sns.color_palette("viridis", n_colors=len(rois) + 1)
        )
    }


def figure0(gs=None):
    from matplotlib.patches import ConnectionPatch

    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 1}):
        if gs is None:
            figure(figsize=(7.5, 4))
            gs = matplotlib.gridspec.GridSpec(
                2,
                6,
                height_ratios=[1, 1.5],
                width_ratios=[1, 0.25, 1, 1, 1, 1],
                hspace=0.1,
                wspace=0.00,
            )
        else:
            gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
                2,
                6,
                gs,
                height_ratios=[1, 1.5],
                width_ratios=[1, 0.25, 1, 1, 1, 1],
                hspace=0.1,
                wspace=0.00,
            )
        ax = subplot(gs[1, :])

        def set(low, high, value=1, set_nan=False, x=None):
            if x is None:
                x = time
            out = x * 0
            out[(low <= x) & (x <= high)] = value
            if set_nan:
                idx = ~(out == value)
                diff = np.where(np.diff(idx))[0]
                idx[diff[0]] = False
                idx[diff[1] + 1] = False
                out[idx] = nan
            return out

        yticks = [
            (5, ""),  #'Reference\ncontrast'),
            (4, "Delay"),
            (5, "Contrast"),
            (3, "Reaction time"),
            (2, "Feedback delay"),
            (1, "Audio feedback"),
        ]
        time = np.linspace(-1.1, 2.1, 5000)
        ref_time = np.linspace(-1.1, -0.1, 5000)
        stim_time = np.linspace(-0.1, 2.1, 5000)
        # Reference
        ax.plot(ref_time, 5 + set(-0.8, -0.4, 0.5, x=ref_time), "k", zorder=5)
        ax.text(-0.35, 5.25, "0.4s", va="center")
        # Reference delay
        ax.plot(time, 4 + set(-0.4, -0.0, 0.5, set_nan=True), ":k", zorder=5)
        #ax.plot(time, 4 + set(-100, -200.0, 0.5, set_nan=False), "k", zorder=5)
        ax.plot([time.min(), -0.4], [4, 4], "k", zorder=5)
        ax.plot([0, time.max()], [4, 4], "k", zorder=5)
        ax.text(0.05, 4.25, "1-1.5s", va="center")
        # Test stimulus
        cvals = array([0.71, 0.33, 0.53, 0.75, 0.59, 0.57, 0.55, 0.61, 0.45, 0.58])
        colors = sns.color_palette(n_colors=10)
        for i in range(10):
            if i == 0:
                ax.plot(
                    stim_time,
                    5
                    + set(0.1 * i, 0.1 * (i + 1), cvals[i], set_nan=True, x=stim_time),
                    color=colors[i],
                    zorder=10,
                )
                ax.plot(
                    stim_time,
                    5
                    + set(0.1 * i, 0.1 * (i + 1), cvals[i], set_nan=False, x=stim_time),
                    "k",
                    zorder=5,
                )
            else:
                ax.plot(
                    stim_time,
                    5
                    + set(0.1 * i, 0.1 * (i + 1), cvals[i], set_nan=True, x=stim_time),
                    color=colors[i],
                )
        ax.text(1.05, 5.25, "100ms/sample", va="center")
        # Response
        ax.plot(time, 3 + set(1, 1.45, 0.5, set_nan=True), ":k", zorder=5)
        ax.plot([time.min(), 1], [3, 3], "k", zorder=5)
        ax.plot([1.45, time.max()], [3, 3], "k", zorder=5)
        ax.text(1.5, 3.25, "0.45s avg", va="center")
        # Feedback delay
        ax.plot(time, 2 + set(1.45, 1.65, 0.5), ":k", zorder=5)
        #ax.plot(time, 2 + set(100.35, 100.55, 0.5), "k", zorder=5)
        ax.text(1.7, 2.25, "0-1.5s", va="center")
        ax.plot([time.min(), 1.45], [2, 2], "k", zorder=5)
        ax.plot([1.65, time.max()], [2, 2], "k", zorder=5)
        # Feedback
        ax.plot(time, 1 + set(1.65, 1.65 + 0.25, 0.5), "k", zorder=5)
        ax.text(1.95, 1.25, "0.25s", va="center")

        ax.set_yticks([])  # i[0] for i in yticks])
        # ax.set_yticklabels([i[1] for i in yticks], va='bottom')
        for y, t in yticks:
            ax.text(-1.1, y + 0.35, t, verticalalignment="center")
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_xticks([])
        ax.tick_params(axis=u"both", which=u"both", length=0)
        ax.set_xlim(-1.1, 2.1)
        ax.set_ylim(0.5, 6.6)

        subax = subplot(gs[0, 0])

        height = 0.25
        pad = 0.1
        overlap = height / 2
        # subax = plt.gcf().add_axes([pad, 1-height-pad, height, height])
        subax.set_xticks([])
        subax.set_yticks([])
        img = make_stimulus(0.5, ringwidth=8 / 4)[:, 400:-400]
        aspect = img.shape[0] / img.shape[1]
        subax.imshow(
            img, aspect="equal", cmap="gray", vmin=0, vmax=1, extent=[0, 1, 0, aspect]
        )
        plt.setp(subax.spines.values(), color="k", linewidth=2)
        xyA = (-0.8, 5.5)  # in axes coordinates
        xyB = (0, 0)  # x in axes coordinates, y in data coordinates
        con = ConnectionPatch(xyA=xyA, xyB=xyB, axesA=ax, coordsA="data", axesB=subax)
        con.set_linewidth(0.5)
        con.set_color([0.5, 0.5, 0.5])
        ax.add_artist(con)

        xyA = (-0.4, 5.5)  # in axes coordinates
        xyB = (1, 0)  # x in axes coordinates, y in data coordinates
        con = ConnectionPatch(xyA=xyA, xyB=xyB, axesA=ax, coordsA="data", axesB=subax)
        con.set_linewidth(0.5)
        con.set_color([0.5, 0.5, 0.5])
        ax.add_artist(con)

        subax = subplot(gs[0, 2])
        subax.set_xticks([])
        subax.set_yticks([])
        img = make_stimulus(cvals[0], ringwidth=8 / 4)[:, 400:-400]
        subax.imshow(
            img, aspect="equal", cmap="gray", vmin=0, vmax=1, extent=[0, 1, 0, aspect]
        )
        # sns.despine(ax=subax, left=True, bottom=True)
        plt.setp(subax.spines.values(), color=colors[0], linewidth=2)

        xyA = (0, 5 + cvals[0])  # in axes coordinates
        xyB = (0, 0)  # x in axes coordinates, y in data coordinates
        con = ConnectionPatch(xyA=xyA, xyB=xyB, axesA=ax, coordsA="data", axesB=subax)
        con.set_linewidth(0.5)
        con.set_color([0.5, 0.5, 0.5])
        ax.add_artist(con)

        subax = subplot(gs[0, 3])
        subax.set_xticks([])
        subax.set_yticks([])
        img = make_stimulus(cvals[1], ringwidth=8 / 4)[:, 400:-400]
        subax.imshow(
            img, aspect="equal", cmap="gray", vmin=0, vmax=1, extent=[0, 1, 0, aspect]
        )
        # sns.despine(ax=subax, left=True, bottom=True)
        plt.setp(subax.spines.values(), color=colors[1], linewidth=2)

        subax = subplot(gs[0, 4])
        subax.set_xticks([])
        subax.set_yticks([])
        img = make_stimulus(cvals[2], ringwidth=8 / 4)[:, 400:-400]
        subax.imshow(
            img, aspect="equal", cmap="gray", vmin=0, vmax=1, extent=[0, 1, 0, aspect]
        )
        # sns.despine(ax=subax, left=True, bottom=True)
        plt.setp(subax.spines.values(), color=colors[2], linewidth=2)

        subax = subplot(gs[0, 5])
        subax.set_xticks([])
        subax.set_yticks([])
        img = make_stimulus(cvals[3], ringwidth=8 / 4)[:, 400:-400]
        subax.imshow(
            img, aspect="equal", cmap="gray", vmin=0, vmax=1, extent=[0, 1, 0, aspect]
        )
        # sns.despine(ax=subax, left=True, bottom=True)
        plt.setp(subax.spines.values(), color=colors[3], linewidth=2)

        xyA = (0.4, 5 + cvals[3])  # in axes coordinates
        xyB = (1, 0)  # x in axes coordinates, y in data coordinates
        con = ConnectionPatch(xyA=xyA, xyB=xyB, axesA=ax, coordsA="data", axesB=subax)
        con.set_linewidth(0.5)
        con.set_color([0.5, 0.5, 0.5])
        ax.add_artist(con)
    return img


def figure1(data=None):
    from conf_analysis.behavior import empirical
    from conf_analysis import behavior

    color_palette = behavior.parse(behavior.colors)
    if data is None:
        data = empirical.load_data()
    data.loc[:, "choice"] = (data.response + 1) / 2
    data.loc[:, "pconf"] = data.confidence - 1

    figure(figsize=(7.5, 7))
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 1.25], hspace=0.05)
    figure0(gs[0, 0])
    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 0.5}):
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            2, 6, gs[1, 0], wspace=1.5, hspace=0.35
        )
        # gs = matplotlib.gridspec.GridSpec(2, 6, wspace=1.5, hspace=0.5)
        def by_discrim(sd, abs=False):
            cvals = np.stack(sd.contrast_probe)
            threshold = (
                (cvals[sd.side == 1].mean(1).mean() - 0.5)
                - (cvals[sd.side == -1].mean(1).mean() - 0.5)
            ) / 2
            if not abs:
                edges = np.linspace(
                    (0.5 + threshold) - 4 * threshold,
                    (0.5 + threshold) + (2 * threshold),
                    11,
                )
                centers = np.around(
                    ((edges[:-1] + np.diff(edges)[0] / 2) - (0.5 + threshold))
                    / threshold,
                    2,
                )
                d = (
                    sd.groupby([pd.cut(sd.mc.values, edges, labels=centers)])
                    .mean()
                    .loc[:, ["choice", "pconf", "mc", "correct"]]
                    .reset_index()
                )
            else:
                edges = np.linspace(
                    (0.5 + threshold) - 1 * threshold,
                    (0.5 + threshold) + (2 * threshold),
                    7,
                )
                centers = np.around(
                    ((edges[:-1] + np.diff(edges)[0] / 2) - (0.5 + threshold))
                    / threshold,
                    2,
                )
                sd.loc[:, "mc"] = np.abs(sd.mc - 0.5) + 0.5
                d = (
                    sd.groupby([pd.cut(sd.mc.values, edges, labels=centers)])
                    .mean()
                    .loc[:, ["choice", "pconf", "mc", "correct"]]
                    .reset_index()
                )
            d.columns = ["threshold_units", "choice", "pconf", "mc", "accuracy"]
            k = d.threshold_units
            d.loc[:, "threshold_units"] = d.loc[:, "threshold_units"].astype(float) + 1
            return d

        subplot(gs[1, :2])
        # Make Panel A from Sanders 2016: Confidence x, vs. accuracy
        # Can't really do that: We don't have fine grained confidence judgements.
        # So have to group by high/low
        X = pd.pivot_table(index="snum", columns="pconf", values="correct", data=data)
        mean = X.mean(0)

        for i in X.index.values:
            plot([0, 1], X.loc[i, :], color="k", alpha=0.25, lw=0.5)

        sem = 2 * X.std(0) / (15 ** 0.5)
        plot([0], mean[0], "o", color=color_palette["Secondary2"][0])
        plot([1], mean[1], "o", color=color_palette["Secondary1"][0])
        plot(
            [0, 0], [sem[0] + mean[0], mean[0] - sem[0]], color_palette["Secondary2"][0]
        )
        plot(
            [1, 1], [sem[1] + mean[1], mean[1] - sem[1]], color_palette["Secondary1"][0]
        )
        xticks([0, 1], [r"Low", r"High"])
        ylabel("% correct")
        xlabel("Confidence")
        xlim(-0.2, 1.2)
        from scipy.stats import ttest_rel, ttest_1samp

        print("T-Test for accuracy by confidence:", ttest_rel(X.loc[:, 0], X.loc[:, 1]))
        sns.despine(ax=gca())
        subplot(gs[1, 2:4])

        dz = (
            data.groupby(["snum", "correct"])
            .apply(lambda x: by_discrim(x, abs=True))
            .reset_index()
        )
        dz.loc[:, "Evidence discriminability"] = dz.threshold_units
        crct = pd.pivot_table(
            index="snum",
            columns="Evidence discriminability",
            values="pconf",
            data=dz.query("correct==1.0"),
        )
        err = pd.pivot_table(
            index="snum",
            columns="Evidence discriminability",
            values="pconf",
            data=dz.query("correct==0.0"),
        )
        from scipy.stats import linregress

        scrct, serr = [], []
        for s in range(1, 16):
            scrct.append(linregress(np.arange(5), crct.loc[s, :2.3])[0])
            serr.append(linregress(np.arange(5), err.loc[s, :2.3])[0])
        print(
            "Slopes for confidence vs. evidence (correct: mean, t, p, #>0):",
            np.mean(scrct),
            ttest_1samp(scrct, 0),
            sum(np.array(scrct) > 0),
        )
        print(
            "Slopes for confidence vs. evidence (error: mean, t, p, #<0):",
            np.mean(serr),
            ttest_1samp(serr, 0),
            sum(np.array(serr) < 0),
        )
        plotwerr(crct, color="g", lw=2, label="Correct")
        plotwerr(err, color="r", lw=2, label="Error")
        legend(frameon=False)

        xticks([1, 2], [r"t", r"2t"])
        xlabel("Evidence discriminability")
        ylabel("Confidence")
        yticks([0.2, 0.3, 0.4, 0.5, 0.6])
        sns.despine(ax=gca())

        subplot(gs[1, 4:])
        dz = (
            data.groupby(["snum", "confidence"])
            .apply(lambda x: by_discrim(x, abs=True))
            .reset_index()
        )
        dz.loc[:, "Evidence discriminability"] = dz.threshold_units
        high = pd.pivot_table(
            index="snum",
            columns="Evidence discriminability",
            values="accuracy",
            data=dz.query("pconf==1.0"),
        )
        et = high.loc[:, :2.2].columns.values

        plotwerr(
            high,
            color=color_palette["Secondary1"][0],
            alpha=1,
            lw=2,
            label="High confidence",
        )
        low = pd.pivot_table(
            index="snum",
            columns="Evidence discriminability",
            values="accuracy",
            data=dz.query("pconf==0.0"),
        )
        plotwerr(
            low,
            color=color_palette["Secondary2"][0],
            alpha=1,
            lw=2,
            label="Low confidence",
        )
        hslope, lslope = [], []
        for s in range(1, 16):
            hslope.append(linregress(et, high.loc[s, :2.2])[0])
            lslope.append(linregress(et, low.loc[s, :2.2])[0])
        print(
            "Slopes for high confidence vs. evidence (mean, t, p, #>0):",
            np.mean(hslope),
            ttest_1samp(hslope, 0),
            sum(np.array(hslope) > 0),
        )
        print(
            "Slopes for low confidence vs. evidence (error: mean, t, p, #<0):",
            np.mean(lslope),
            ttest_1samp(lslope, 0),
            sum(np.array(lslope) < 0),
        )
        print("High vs low slope:", ttest_rel(hslope, lslope))
        legend(frameon=False)
        xticks([1, 2], [r"t", r"2t"])
        xlabel("Evidence discriminability")
        ylabel("% correct")
        yticks([0.5, 0.75, 1], [50, 75, 100])
        sns.despine(ax=gca())
        # tight_layout()

        ax = subplot(gs[0, :3])

        palette = {
            r"$E_{N}^{High}$": color_palette["Secondary1"][0],
            r"$E_{N}^{Low}$": color_palette["Secondary2"][1],
            r"$E_{S}^{Low}$": color_palette["Secondary2"][1],
            r"$E_{S}^{High}$": color_palette["Secondary1"][0],
        }

        k = empirical.get_confidence_kernels(data, contrast_mean=0.5)
        for kernel, kdata in k.groupby("Kernel"):
            kk = pd.pivot_table(
                index="snum", columns="time", values="contrast", data=kdata
            )
            plotwerr(kk, color=palette[kernel], lw=2, label="Low confidence")
        # empirical.plot_kernel(k, palette, legend=False)
        plt.ylabel(r"$\Delta$ Contrast")
        plt.text(
            -0.2,
            0.003,
            "ref.    test",
            rotation=90,
            horizontalalignment="center",
            verticalalignment="center",
        )
        xlabel("Contrast sample")
        # legend(frameon=False)
        xticks(np.arange(10), np.arange(10) + 1)
        ax.axhline(color="k", lw=1)
        ax.set_ylim(-0.04, 0.04)
        sns.despine(ax=ax)

        ax = subplot(gs[0, 3:])
        from conf_analysis.behavior import kernels
        from conf_analysis.meg import cort_kernel as ck

        dk, confidence_kernel = ck.get_kernels()
        scrct, serr = [], []
        for s in range(1, 16):
            scrct.append(linregress(np.arange(10), dk.loc[s,])[0])
            serr.append(linregress(np.arange(10), confidence_kernel.loc[s,])[0])
        print(
            "Slopes choice kernel (correct: mean, t, p, #>0):",
            np.mean(scrct),
            ttest_1samp(scrct, 0),
            sum(np.array(scrct) > 0),
        )
        print(
            "Slopes for confidence kernel (error: mean, t, p, #<0):",
            np.mean(serr),
            ttest_1samp(serr, 0),
            sum(np.array(serr) < 0),
        )

        plotwerr(dk, color="k", label="Decision kernel")
        draw_sig(ax, dk, fdr=True, color="k", y=0.0)
        plotwerr(confidence_kernel, color=(0.5, 0.5, 0.5), label="Confidence kernel")
        draw_sig(ax, confidence_kernel, fdr=True, color=(0.5, 0.5, 0.5), y=0.005)
        # plotwerr(confidence_assym+0.5, 'b--', label='Confidence assym.')

        # axhline(0.5, color="k", lw=1)
        ylim([0.49 - 0.5, 0.64 - 0.5])
        ylabel("AUC-0.5")
        xlabel("Contrast sample")
        legend(frameon=False)
        yticks(np.array([0.5, 0.55, 0.6]) - 0.5)
        xticks(np.arange(10), np.arange(10) + 1)
        sns.despine(ax=gca(), bottom=False)
        savefig(
            "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/1_figure_1.pdf",
            bbox_inches="tight",
            dpi=1200,
        )

        p = data.groupby(["snum"]).apply(
            lambda x: empirical.fit_choice_logistic(x, summary=False)
        )
        p = p.groupby("snum").mean()
        print(
            "Can predict choice above chance in %i/15 subjects. \nMean choice accuracy is %0.2f"
            % ((sum(p > 0.5)), np.around(np.mean(p), 2))
        )

        p = data.groupby(["snum", "response"]).apply(
            lambda x: empirical.fit_conf_logistic(x, summary=False)
        )
        p = p.groupby("snum").mean()
        print(
            "Can predict confidence above chance in %i/15 subjects. \nMean confidence accuracy is %0.2f"
            % ((sum(p > 0.5)), np.around(np.mean(p), 2))
        )
    return ck


def figure2(df=None, stats=False):
    if not stats:
        import gzip, pickle

        with gzip.open(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_stats_20190516.pickle",
            "rb",
        ) as f:
            stats = pickle.load(f)

    if df is None:
        df = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_confmeg-20190516.hdf"
        )
    palette = _stream_palette()
    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 0.5}):
        figure(figsize=(7.5, 7.5 / 2))
        from conf_analysis.meg import srtfr

        # gs = matplotlib.gridspec.GridSpec(3, 2, width_ratios=[0.99, 0.01])

        fig = srtfr.plot_stream_figures(
            df.query('hemi=="avg"'),
            contrasts=["all"],
            flip_cbar=False,
            # gs=gs[0, 0],
            stats=stats,
            title_palette=palette,
        )

        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-50, 50), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        # [left, bottom, width, height]
        cax = fig.add_axes([0.74, 0.2, 0.1, 0.015])
        cb = colorbar(
            cmap,
            cax=cax,
            shrink=0.5,
            orientation="horizontal",
            ticks=[-50, 0, 50],
            drawedges=False,
            label="% Power\nchange",
        )
        cb.outline.set_visible(False)
        sns.despine(ax=cax)

        view_one = dict(azimuth=-40, elevation=100, distance=350)
        view_two = dict(azimuth=-145, elevation=70, distance=350)
        img = _get_palette(palette, views=[view_two, view_one])
        iax = fig.add_axes([0.09, 0.75, 0.25, 0.25])
        iax.imshow(img)
        iax.set_xticks([])
        iax.set_yticks([])
        sns.despine(ax=iax, left=True, bottom=True)

    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/2_figure_2.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    return img, df, stats


def figure3(df=None, stats=False, dcd=None, aspect="auto"):  # 0.01883834992799947):
    """
    Plot TFRs underneath each other.
    """
    if not stats:
        import gzip, pickle

        with gzip.open(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_confmeg-20190308-stats.pickle.gzip",
            "rb",
        ) as f:
            stats = pickle.load(f)

    if df is None:
        df = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_confmeg-20190308.hdf"
        )

    if dcd is None:
        dcd = dp.get_decoding_data()
    palette = _stream_palette()

    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 0.5}):
        fig = figure(figsize=(8, 7))
        from conf_analysis.meg import srtfr

        gs = matplotlib.gridspec.GridSpec(2, 1)

        srtfr.plot_stream_figures(
            df.query('~(hemi=="avg")'),
            contrasts=["choice"],
            flip_cbar=True,
            gs=gs[0, 0],
            stats=stats,
            aspect=aspect,
            title_palette=palette,
        )

        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-25, 25), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        cax = fig.add_axes([0.74, 0.57, 0.1, 0.0125 / 2])
        cb = colorbar(
            cmap,
            cax=cax,
            shrink=0.5,
            ticks=[-25, 0, 25],
            drawedges=False,
            orientation="horizontal",
            label="% change",
        )
        cb.outline.set_visible(False)

        plotter = dp.StreamPlotter(
            dp.plot_config,
            {"MIDC_split": "Reds", "CONF_unsigned": "Greens", "CONF_signed": "Blues"},
            {
                # "Averaged": df.test_roc_auc.Averaged,
                "Lateralized": dcd.test_roc_auc.Lateralized
            },
            gs=gs[1, 0],
            title_palette=palette,
        )
        plotter.plot(aspect="auto")

        cax = fig.add_axes([0.81, 0.21, 0.1, 0.0125 / 2])
        cax.plot([-10, -1], [0, 0], "r", label="Choice")
        cax.plot([-10, -1], [0, 0], "b", label="Signed confidence")
        cax.plot([-10, -1], [0, 0], "g", label="Unigned confidence")
        cax.set_xlim([0, 1])
        cax.set_xticks([])
        cax.set_yticks([])
        cax.legend(frameon=False)
        sns.despine(ax=cax, left=True, bottom=True)

    # plt.tight_layout()
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/3_figure_3.pdf"
    )
    return df, stats, dcd


@memory.cache()
def ct(x):
    from mne.stats import permutation_cluster_1samp_test as cluster_test

    return cluster_test(
        x,
        threshold={"start": 0, "step": 0.2},
        connectivity=None,
        tail=0,
        n_permutations=1000,
        n_jobs=4,
    )[2]


def figure_3_test_sig(dcd):
    for cluster in dcd.test_roc_auc.Lateralized.columns:
        if cluster.startswith("NSW"):
            continue
        choice = pd.pivot_table(
            dcd.test_roc_auc.Lateralized.query(
                'epoch=="stimulus" & signal=="MIDC_split"'
            ),
            index="subject",
            columns="latency",
            values=cluster,
        ).loc[:, 0:1.1]
        unconf = pd.pivot_table(
            dcd.test_roc_auc.Lateralized.query(
                'epoch=="stimulus" & signal=="CONF_signed"'
            ),
            index="subject",
            columns="latency",
            values=cluster,
        ).loc[:, 0:1.1]
        siconf = pd.pivot_table(
            dcd.test_roc_auc.Lateralized.query(
                'epoch=="stimulus" & signal=="CONF_unsigned"'
            ),
            index="subject",
            columns="latency",
            values=cluster,
        ).loc[:, 0:1.1]
        u = (choice - unconf).values
        s = (choice - siconf).values
        upchoice = ct(u)
        spchoice = ct(s)
        if (sum(upchoice < 0.05) > 0) or (sum(spchoice < 0.05) > 0):
            print("----->", cluster, upchoice.shape)
            print(
                cluster,
                "Choice vs. unsigned #sig:",
                sum(upchoice < 0.05),
                "#Conf larger:",
                np.sum(u.mean(0)[upchoice < 0.05] < 0),
            )
            print(
                cluster,
                " Choice vs signed #sig:",
                sum(spchoice < 0.05),
                "#Conf larger:",
                np.sum(s.mean(0)[spchoice < 0.05] < 0),
            )


def figure3_supplement(df=None, stats=None):
    if not stats:
        import gzip, pickle

        with gzip.open(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_confmeg-20190308-stats.pickle.gzip",
            "rb",
        ) as f:
            stats = pickle.load(f)

    if df is None:
        df = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_contrasts_confmeg-20190308.hdf"
        )

    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 0.5}):
        figure(figsize=(7.5, 7.5 / 2))
        from conf_analysis.meg import srtfr

        # gs = matplotlib.gridspec.GridSpec(3, 2, width_ratios=[0.99, 0.01])

        fig = srtfr.plot_stream_figures(
            df.query('hemi=="avg"'),
            contrasts=["stimulus"],
            flip_cbar=False,
            # gs=gs[0, 0],
            stats=stats,
        )

        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-25, 25), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        # [left, bottom, width, height]
        cax = fig.add_axes([0.74, 0.2, 0.1, 0.015])
        cb = colorbar(
            cmap,
            cax=cax,
            shrink=0.5,
            orientation="horizontal",
            ticks=[-25, 0, 25],
            drawedges=False,
            label="% change",
        )
        cb.outline.set_visible(False)
        sns.despine(ax=cax)

    # fig.colorbar(im, cax=cax)
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/3_figure_3_supplement.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    return fig, df, stats


def figure4(ogldcd=None, pdcd=None):
    if ogldcd is None:
        ogldcd = dp.get_decoding_data(restrict=False, ogl=True)
    if pdcd is None:
        pdcd = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_vert_phase_decoding.hdf"
        )
    plt.figure(figsize=(7.5, 4))
    gs = matplotlib.gridspec.GridSpec(1, 1)  # ,height_ratios=[3 / 5, 2 / 5])
    _figure4A(ogldcd, gs=gs[0, 0])
    tight_layout()
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/4_figure_4.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    plt.figure(figsize=(7.5, 4))
    gs = matplotlib.gridspec.GridSpec(1, 1)  # , height_ratios=[3 / 5, 2 / 5])
    _figure4B(pdcd, gs=gs[0, 0])
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_4_figure_4.pdf",
        dpi=1200,
        bbox_inches="tight",
    )

    return ogldcd, pdcd


fig_4_interesting_rois = {"3b": "M1", "V1": "V1", "2": "IPS/PostCeS", "7PC": "aIPS"}
fig_4B_interesting_rois = {
    "JWG_M1": "M1",
    "vfcPrimary": "V1",
    "JWG_IPS_PCeS": "IPS/PostCeS",
    "JWG_aIPS": "aIPS",
}


def _figure4A(data=None, t=1.083, gs=None):
    import seaborn as sns
    import re
    import matplotlib

    if data is None:
        data = dp.get_decoding_data(restrict=False, ogl=True)
    df = data.test_roc_auc.Pair
    df = df.query('epoch=="stimulus"')

    palettes = {"respones": {}, "unsigned_confidence": {}, "signed_confidence": {}}
    colormaps = {
        "MIDC_split": "Reds",
        "CONF_signed": "Blues",
        "CONF_unsigned": "Greens",
    }
    titles = {
        "MIDC_split": "Choice",
        "CONF_signed": "Signed confidence",
        "CONF_unsigned": "Unsigned confidence",
    }
    v_values = {
        "MIDC_split": (0.5, 0.7),
        "CONF_signed": (0.5, 0.6),
        "CONF_unsigned": (0.5, 0.55),
    }
    high = 0.65
    low = 0.5
    if gs is None:
        plt.figure(figsize=(7.5, 5))
        gs = matplotlib.gridspec.GridSpec(4, 3)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            4, 3, subplot_spec=gs, hspace=0.0, height_ratios=[1, 0.5, 1, 1]
        )

    def corrs(data, t):
        res = {}
        data = data.query("latency==%f" % (t))
        choice = pd.pivot_table(
            index="subject",
            columns="cluster",
            values=0,
            data=data.query('signal=="%s"' % "MIDC_split").stack().reset_index(),
        )
        uns = pd.pivot_table(
            index="subject",
            columns="cluster",
            values=0,
            data=data.query('signal=="%s"' % "CONF_unsigned").stack().reset_index(),
        )
        sig = pd.pivot_table(
            index="subject",
            columns="cluster",
            values=0,
            data=data.query('signal=="%s"' % "CONF_signed").stack().reset_index(),
        )
        res["Ch./Un."] = [
            np.corrcoef(choice.loc[i, :], uns.loc[i, :])[0, 1] for i in range(1, 16)
        ]
        res["Ch./Si."] = [
            np.corrcoef(choice.loc[i, :], sig.loc[i, :])[0, 1] for i in range(1, 16)
        ]
        res["Si./Un."] = [
            np.corrcoef(sig.loc[i, :], uns.loc[i, :])[0, 1] for i in range(1, 16)
        ]
        return pd.DataFrame(res)

    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 1}):
        for i, signal in enumerate(colormaps.keys()):
            low, high = v_values[signal]
            # for i, (signal, data) in enumerate(df.groupby(["signal"])):
            sdata = df.query('signal=="%s"' % signal)
            plt.subplot(gs[0, i], zorder=-i)
            X = pd.pivot_table(
                index="cluster",
                columns="latency",
                values=0,
                data=sdata.stack().reset_index(),
            )

            txts = []
            idx = np.argsort(X.loc[:, t])
            sorting = X.index.values[idx]
            norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
            cm = matplotlib.cm.get_cmap(colormaps[signal])
            max_roi = X.loc[:, 1.417].idxmax()
            for j, roi in enumerate(sorting):
                val, color = X.loc[roi, t], cm(norm(X.loc[roi, t]))
                plt.plot(X.columns.values, X.loc[roi, :], color=color)
                if any([roi == x for x in fig_4_interesting_rois.keys()]) or (
                    roi == max_roi
                ):
                    try:
                        R = fig_4_interesting_rois[roi]
                    except KeyError:
                        R = roi
                    txts.append(plt.text(1.417, X.loc[roi, 1.417] - 0.005, R))
            y = np.linspace(0.475, 0.75, 200)
            x = y * 0 - 0.225
            plt.scatter(x, y, c=cm(norm(y)), marker=0)
            plt.title(titles[signal])
            plt.xlim([-0.25, 1.4])
            plt.ylim([0.475, 0.75])
            plt.axvline(1.2, color="k", alpha=0.9)
            plt.xlabel("Time", zorder=1)
            if i > 0:
                sns.despine(ax=plt.gca(), left=True)
                plt.yticks([])
            else:
                plt.ylabel("AUC")
                sns.despine(ax=plt.gca())

            palette = {
                d.replace("dlpfc_", "").replace("pgACC_", ""): X.loc[d, t]
                for d in X.index.values
            }
            # print(palette)
            img = _get_lbl_annot_img(
                palette,
                low=low,
                high=high,
                views=[["par", "front"], ["med", "lat"]],
                colormap=colormaps[signal],
            )

            plt.subplot(gs[2, i], aspect="equal", zorder=-10)
            plt.imshow(img, zorder=-10)
            plt.xticks([])
            plt.yticks([])
            sns.despine(ax=plt.gca(), left=True, bottom=True)
        ax = plt.subplot(gs[3, :-1])
        from matplotlib.cm import get_cmap
        from matplotlib.colors import Normalize

        def foo(k):
            norm = Normalize(*v_values[k])
            cmap = get_cmap(colormaps[k])

            def bar(x):
                return cmap(norm(x))

            return bar

        colors = {k: foo(k) for k in colormaps.keys()}
        idx = dp._plot_signal_comp(
            df,
            t,
            None,
            colors,
            "Pair",
            auc_cutoff=0.5,
            ax=ax,
            horizontal=True,
            plot_labels=False,
            color_by_cmap=True,
        )
        signals, subjects, areas, _data = dp.get_signal_comp_data(df, t, "stimulus")
        choice = _data[0, :]
        unsign = _data[1, :]
        signed = _data[2, :]
        _t, _p = ttest_rel(choice.T, unsign.T)
        print("# of Rois where unsigned>choice:", np.sum(_t[_p < 0.05] < 0))
        _t, _p = ttest_rel(choice.T, signed.T)
        print("# of Rois where signed>choice:", np.sum(_t[_p < 0.05] < 0))
        ax.set_xlabel("ROI")
        ax.set_ylabel("AUC")
        ax.set_title("")
        ax.set_ylim([0.49, 0.7])
        y = np.linspace(0.49, 0.7, 250)
        x = y * 0 - 3
        sns.despine(ax=ax, bottom=True)
        o = corrs(df, t=t).stack().reset_index()
        o.columns = ["idx", "Comparison", "Correlation"]
        # o.Comparison.replace({'Ch./Un.':r'\textcolor{red}{Ch./Un.}'}, inplace=True)

        ax = plt.subplot(gs[3, -1])
        sns.stripplot(
            x="Comparison",
            y="Correlation",
            color="k",
            alpha=0.75,
            dodge=True,
            jitter=True,
            ax=ax,
            data=o,
            order=["Ch./Si.", "Ch./Un.", "Si./Un."],
        )
        ax.set_xticklabels(
            ["Choice vs\nSigned", "Choice vs.\nUnsigned", "Signed vs.\nUnsigned"]
        )
        ax.set_xlabel("")
        print(
            "Corr, choice vs signed:",
            o.query('Comparison=="Ch./Si."').Correlation.mean(),
            ttest_1samp(o.query('Comparison=="Ch./Si."').Correlation, 0),
        )
        print(
            "Corr, choice vs unsigned:",
            o.query('Comparison=="Ch./Un."').Correlation.mean(),
            ttest_1samp(o.query('Comparison=="Ch./Un."').Correlation, 0),
        )
        print(
            "Corr, signed vs unsigned:",
            o.query('Comparison=="Si./Un."').Correlation.mean(),
            ttest_1samp(o.query('Comparison=="Si./Un."').Correlation, 0),
        )
        y = o.query('Comparison=="Ch./Si."').Correlation.mean()
        plot([-0.15, 0.15], [y, y], "k")
        y = o.query('Comparison=="Ch./Un."').Correlation.mean()
        plot([1 - 0.15, 1 + 0.15], [y, y], "k")
        y = o.query('Comparison=="Si./Un."').Correlation.mean()
        plot([2 - 0.15, 2 + 0.15], [y, y], "k")
        sns.despine(ax=ax, bottom=True)
    tight_layout()
    return data


def _figure4B(df=None, gs=None, t=1.1):
    import seaborn as sns
    import re
    import matplotlib

    if df is None:
        df = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_vert_phase_decoding.hdf"
        )

    colormaps = {
        "response": "Reds",
        "signed_confidence": "Blues",
        "unsigned_confidence": "Greens",
    }
    titles = {
        "response": "Choice",
        "signed_confidence": "Signed confidence",
        "unsigned_confidence": "Unsigned confidence",
    }
    v_values = {
        "response": (0.5, 0.7),
        "signed_confidence": (0.5, 0.6),
        "unsigned_confidence": (0.5, 0.55),
    }
    if gs is None:
        plt.figure(figsize=(7.5, 5))
        gs = matplotlib.gridspec.GridSpec(2, 3)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=gs, hspace=0.3
        )
    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 1}):
        for i, (signal, data) in enumerate(df.groupby(["target"])):
            low, high = v_values[signal]
            plt.subplot(gs[0, i], zorder=-i)
            X = pd.pivot_table(
                index="roi", columns="latency", values="test_roc_auc", data=data
            )

            txts = []
            idx = np.argsort(X.loc[:, t])
            sorting = X.index.values[idx]
            norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
            cm = matplotlib.cm.get_cmap(colormaps[signal])

            for j, roi in enumerate(sorting):
                val, color = X.loc[roi, t], cm(norm(X.loc[roi, t]))
                plt.plot(X.columns.values, X.loc[roi, :], color=color)
                if any([roi == x for x in fig_4B_interesting_rois.keys()]):
                    R = fig_4B_interesting_rois[roi]
                    txts.append(plt.text(1.4, X.loc[roi, 1.4], R))
            y = np.linspace(0.475, 0.75, 200)
            x = y * 0 - 0.225
            plt.scatter(x, y, c=cm(norm(y)), marker=0)
            plt.title(titles[signal])
            plt.xlim([-0.25, 1.4])
            plt.ylim([0.475, 0.75])
            plt.xlabel("Time")
            plt.axvline(t, color="k", alpha=0.9)
            if i > 0:
                sns.despine(ax=plt.gca(), left=True)
                plt.yticks([])
            else:
                plt.ylabel("AUC")
                sns.despine(ax=plt.gca())

            palette = {
                d.replace("dlpfc_", "").replace("pgACC_", ""): X.loc[d, t]
                for d in X.index.values
            }

            img = _get_lbl_annot_img(
                palette,
                low=low,
                high=high,
                views=[["par", "front"], ["med", "lat"]],
                colormap=colormaps[signal],
            )
            plt.subplot(gs[1, i], aspect="equal", zorder=-10)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            sns.despine(ax=plt.gca(), left=True, bottom=True)

    tight_layout()
    return df


def figure5(
    ssd=None, idx=None, oglssd=None, oglidx=None, brain=None, integration_slice=def_ig
):
    if ssd is None:
        ssd = dp.get_ssd_data(restrict=False)
    if idx is None:
        idx = dp.get_ssd_idx(ssd.test_slope, integration_slice=integration_slice)
    if oglssd is None:
        oglssd = dp.get_ssd_data(restrict=False, ogl=True)
    if oglidx is None:
        oglidx = dp.get_ssd_idx(
            oglssd.test_slope, integration_slice=integration_slice, pair=True
        )
    if "post_medial_frontal" in ssd.test_slope.columns:
        del ssd.test_slope["post_medial_frontal"]
    if "vent_medial_frontal" in ssd.test_slope.columns:
        del ssd.test_slope["vent_medial_frontal"]
    if "ant_medial_frontal" in ssd.test_slope.columns:
        del ssd_test_slope["ant_medial_frontal"]
    plt.figure(figsize=(8, 10))
    gs = matplotlib.gridspec.GridSpec(
        9, 3, height_ratios=[0.8, 0.4, 0.5, 0.25, 0.8, 0.35, 0.5, 0.45, 0.8], hspace=0.0
    )
    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 1}):
        ax = subplot(gs[0, 0])
        plot_per_sample_resp(
            plt.gca(), ssd.test_slope.Averaged, "vfcPrimary", "V1", integration_slice
        )
        legend([], frameon=False)
        ax = subplot(gs[0, 1])
        plot_per_sample_resp(
            plt.gca(),
            ssd.test_slope.Lateralized,
            "JWG_IPS_PCeS",
            "IPS/PostCeS",
            integration_slice,
            [-0.005, 0.065],
        )
        ax.set_ylabel("")
        legend([], frameon=False)

        ax = subplot(gs[0, 2])
        plot_per_sample_resp(
            plt.gca(),
            ssd.test_slope.Lateralized,
            "JWG_M1",
            "M1 (hand)",
            integration_slice,
            [-0.005, 0.065],
        )
        ax.set_ylabel("")

        legend(frameon=False)
        sns.despine()

        _figure5A(oglssd, oglidx, gs[2, :])
        _figure6A(gs=gs[4, :])
        _figure6B(gs=gs[6, :])
        # _figure5C(ssd.test_slope.Averaged, oglssd.test_slope.Pair, gs=gs[3,:])
        savefig(
            "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/5_figure_5.pdf",
            dpi=1200,
            bbox_inches="tight",
        )

        plt.figure(figsize=(7.5, 1.5))
        gs = matplotlib.gridspec.GridSpec(1, 3, hspace=0.0)
        _figure5B(gs=gs[:, :])
        savefig(
            "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_5_figure_5.pdf",
            dpi=1200,
            bbox_inches="tight",
        )
    return ssd, idx, brain


def _figure5A(ssd, idx, gs, integration_slice=def_ig):
    import seaborn as sns
    import matplotlib

    gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs)
    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 0.5}):
        dv = 10
        ax = plt.subplot(gs[0, 0])
        m = idx.groupby("cluster").mean()
        palette = {k: dv + vals.SSD for k, vals in m.iterrows()}
        img = _get_img(palette, low=dv + -0.05, high=dv + 0.05, views=["lat", "med"])
        plt.imshow(img, aspect="equal")
        plt.xticks([])
        plt.yticks([])
        plt.title("Contrast encoding", fontsize=8)
        sns.despine(ax=ax, left=True, right=True, bottom=True)
        ax = plt.subplot(gs[0, 1])
        m = idx.groupby("cluster").mean()
        palette = {k: dv + vals.SSD_acc_contrast for k, vals in m.iterrows()}
        img = _get_img(palette, low=dv + -0.05, high=dv + 0.05, views=["lat", "med"])
        plt.imshow(img, aspect="equal")
        plt.xticks([])
        plt.yticks([])
        plt.title("Accumulated contrast", fontsize=8)
        sns.despine(ax=ax, left=True, right=True, bottom=True)
        ax = plt.subplot(gs[0, 2])

        m = idx.groupby("cluster").mean()
        palette = {k: dv + vals.SSDvsACC for k, vals in m.iterrows()}
        img = _get_img(palette, low=dv + -0.025, high=dv + 0.025, views=["lat", "med"])
        plt.imshow(img, aspect="equal")
        plt.xticks([])
        plt.yticks([])
        plt.title("Difference", fontsize=8)
        sns.despine(ax=ax, left=True, right=True, bottom=True)

    return ssd, idx


def _figure5B(xscores=None, gs=None):
    from scipy.stats import ttest_rel, ttest_1samp
    from mne.stats import fdr_correction

    if xscores is None:
        import pickle

        xscores = (
            pickle.load(
                open(
                    "/Users/nwilming/u/conf_analysis/results/all_areas_gamma_Xarea_stim_latency.pickle",
                    "rb",
                )
            )["scores"]
            .set_index(["latency", "subject", "motor_area"], append=True)
            .stack()
            .reset_index()
        )
        xscores.columns = [
            "del",
            "latency",
            "subject",
            "motor_area",
            "comparison",
            "corr",
        ]
        xscores = xscores.iloc[:, 1:]

    colors = ["windows blue", "amber", "faded green", "dusty purple"]

    if gs is None:
        figure(figsize=(7.5, 2.5))
        gs = matplotlib.gridspec.GridSpec(1, 5, width_ratios=[1, 0.25, 1, 1, 1])
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=gs, width_ratios=[1, 0.35, 1, 1, 1]
        )
    area_names = ["aIPS", "IPS/PostCeS", "M1 (hand)"]
    area_colors = _stream_palette()
    for i, signal in enumerate(["JWG_aIPS", "JWG_IPS_PCeS", "JWG_M1"]):
        sd = xscores.query('motor_area=="%s" & (0<=latency<=0.2)' % signal)

        yc = pd.pivot_table(
            columns="latency",
            index="subject",
            values="corr",
            data=sd.query('comparison=="corr"'),
        )
        y1smp = pd.pivot_table(
            columns="latency",
            index="subject",
            values="corr",
            data=sd.query('comparison=="1smp_corr"'),
        )
        yint = pd.pivot_table(
            columns="latency",
            index="subject",
            values="corr",
            data=sd.query('comparison=="integrator_corr"'),
        )
        t, p = ttest_rel(yc, y1smp)

        t, pvsnull = ttest_1samp(yc, 0)
        ax = subplot(gs[0, i + 2])

        ax.plot(
            yc.columns.values,
            (yc.values).mean(0),
            "-",
            color=sns.xkcd_rgb[colors[0]],
            label="Ten samples",
        )
        ax.set_ylim([-0.01, 0.05])
        ax.plot(
            y1smp.columns.values,
            (y1smp.values).mean(0),
            "-",
            color=sns.xkcd_rgb[colors[1]],
            label="Last sample",
        )
        ax.plot(
            yint.columns.values,
            (yint.values).mean(0),
            "-",
            color=sns.xkcd_rgb[colors[2]],
            label="Integrator",
        )
        id_cor, _ = fdr_correction(p)  # <0.05
        id_unc = pvsnull < 0.05
        draw_sig(ax, yc - y1smp, fdr=False, lw=2)
        # draw_sig(ax, yc-y1smp, fdr=False, color='g', zorder=0, lw=2)
        # draw_sig(ax, yc, fdr=False, alpha=0.5, lw=1)
        title(area_names[i], fontsize=8, color=area_colors[signal])
        if i == 0:
            ax.set_xlabel("Time after sample onset")
        if i == 0:
            ax.set_ylabel("Correlation")
            ax.legend(frameon=False, loc=9)
            sns.despine(ax=ax)
        else:

            ax.set_yticks([])
            sns.despine(ax=ax, left=True)
    ax = subplot(gs[0, 0])
    # img = _get_lbl_annot_img({'vfcPrimary':0.31, 'JWG_M1':1, 'JWG_aIPS':0.8, 'JWG_IPS_PCeS':0.9},
    #    low=0.1, high=1, views=[["par"]], colormap='viridis')
    img = _get_palette(
        {
            "vfcPrimary": area_colors["vfcPrimary"],
            "JWG_M1": area_colors["JWG_M1"],
            "JWG_aIPS": area_colors["JWG_aIPS"],
            "JWG_IPS_PCeS": area_colors["JWG_IPS_PCeS"],
        },
        views=[["par"]],
    )
    ax.imshow(img, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    from matplotlib import patches

    style = "Simple,tail_width=0.5,head_width=4,head_length=8"

    a3 = patches.FancyArrowPatch(
        (501, 559),
        (295, 151),
        connectionstyle="arc3,rad=.25",
        **dict(arrowstyle=style, color=area_colors["JWG_aIPS"])
    )
    a2 = patches.FancyArrowPatch(
        (501, 559),
        (190, 229),
        connectionstyle="arc3,rad=.15",
        **dict(arrowstyle=style, color=area_colors["JWG_IPS_PCeS"])
    )
    a1 = patches.FancyArrowPatch(
        (501, 559),
        (225, 84),
        connectionstyle="arc3,rad=.55",
        **dict(arrowstyle=style, color=area_colors["JWG_M1"])
    )
    ax.add_patch(a3)
    ax.add_patch(a2)
    ax.add_patch(a1)
    sns.despine(ax=ax, left=True, bottom=True)


def _figure5C(ssd, oglssd, gs=None):
    if gs is None:
        gs = matplotlib.gridspec.GridSpec(1, 4)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=gs, width_ratios=[1, 1, 0.5, 0.5]
        )
    from conf_analysis.behavior import kernels, empirical
    from scipy.stats import ttest_1samp
    from mne.stats import fdr_correction
    import pickle

    # dz = empirical.get_dz()
    data = empirical.load_data()
    K = data.groupby("snum").apply(kernels.get_decision_kernel)
    # K = dp.extract_kernels(data, contrast_mean=0.5, include_ref=True).T
    C = (
        data.groupby("snum")
        .apply(kernels.get_confidence_kernel)
        .stack()
        .groupby("snum")
        .mean()
    )

    ax = subplot(gs[0, -1])
    ax.plot(K.mean(0), color="k", label="Choice")  # -K.mean(0).mean()
    ax.plot(C.mean(0), color=(0.5, 0.5, 0.5), label="Confidence")  # -C.mean(0).mean()
    ax.legend(frameon=False, bbox_to_anchor=(0.3, 1), loc="upper left")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Behavior")
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_title('behavior')
    # ax.set_xticks(np.arange(10))
    # ax.set_xticklabels({k+1:k+1 for k in np.arange(10)})
    sns.despine(ax=ax, left=True)
    ax = subplot(gs[0, -2])
    ex_kernels = pickle.load(
        open(
            "/Users/nwilming/u/conf_analysis/results/example_kernel_vfcPrimary.pickle",
            "rb",
        )
    )
    ex_choice = ex_kernels["choice"]
    ax.plot((ex_choice.mean(0)), color="k", label="Choice")  # -ex_choice.mean()
    ex_conf = ex_kernels["conf"]
    ax.plot(
        (ex_conf.mean(0)), color=(0.5, 0.5, 0.5), label="Confidence"
    )  # ex_conf.mean()
    # ax.legend(frameon=False, bbox_to_anchor= (0.3, 1), loc='upper left')
    # ax.set_title('V1')
    ax.set_xlabel("Sample")
    ax.set_ylabel("V1")
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_xticks(np.arange(10))
    # ax.set_xticklabels({k+1:k+1 for k in np.arange(10)})
    sns.despine(ax=ax, left=True)

    cck = pd.read_hdf(
        "/Users/nwilming/u/conf_analysis/results/choice_kernel_correlations.hdf"
    )
    cm = cck.groupby("cluster").mean()
    K_t_palette = dict(cm.choice_corr)
    C_t_palette = dict(cm.conf_corr)
    ax = subplot(gs[0, 0])
    voffset = 1
    low = -0.6
    high = 0.6
    img = _get_lbl_annot_img(
        {k: v + voffset for k, v in K_t_palette.items()},
        views=["lat", "med"],
        low=voffset + low,
        high=voffset + high,
        thresh=None,
    )
    ax.imshow(img, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Choice kernel correlation")
    sns.despine(ax=ax, left=True, bottom=True)

    ax = subplot(gs[0, 1])
    img = _get_lbl_annot_img(
        {k: v + voffset for k, v in C_t_palette.items()},
        views=["lat", "med"],
        low=voffset + low,
        high=voffset + high,
        thresh=None,
    )
    ax.imshow(img, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Corr. w Conf. kernel")
    sns.despine(ax=ax, left=True, bottom=True)
    return  # K_p_palette, K_t_palette


figure_6colors = sns.color_palette(n_colors=3)
figure_6colors = {
    "AFcorr": figure_6colors[0],
    "DCD": figure_6colors[1],
    "AF-CIFcorr": figure_6colors[2],
    "lAFcorr": figure_6colors[0],
    "lAF-CIFcorr": figure_6colors[2],
}


def _figure6B(cluster="vfcPrimary", gs=None):
    import pickle
    from scipy.stats import linregress, ttest_1samp

    res = []
    clpeaks = get_cl_decoding_peaks()
    oglpeaks = get_ogl_decoding_peaks()
    v1decoding = clpeaks["vfcPrimary"]
    intdecoding = oglpeaks["3b"]
    m1decoding = clpeaks["JWG_M1"]
    corrs = []
    rescorr = []
    for freq in list(range(1, 10)) + list(range(10, 115, 5)):
        fname = (
            "/Users/nwilming/u/conf_analysis/results/ncort_kernel_f%i.results.pickle"
            % freq
        )
        a = pickle.load(open(fname, "rb"))
        ccs, K, kernels, _ = a["ccs"], a["K"], a["kernels"], a["v1decoding"]
        kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
        KK = np.stack(kernels.kernel)
        kernels = pd.DataFrame(KK, index=kernels.index).query('cluster=="%s"' % cluster)
        rems = pd.pivot_table(data=kernels.query("rmcif==True"), index="subject")
        alls = pd.pivot_table(data=kernels.query("rmcif==False"), index="subject")
        allsV1 = alls.mean(1).to_frame()
        allsV1.loc[:, "comparison"] = "V1kernel"
        allsV1.loc[:, "freq"] = freq
        allsV1.columns = ["sum", "comparison", "freq"]
        res.append(allsV1)
        remsV1 = rems.mean(1).to_frame()
        remsV1.loc[:, "comparison"] = "V1kernelCIF"
        remsV1.loc[:, "freq"] = freq
        remsV1.columns = ["sum", "comparison", "freq"]
        res.append(remsV1)
        allsslopes = alls.apply(
            lambda x: linregress(np.arange(10), x)[0], axis=1
        ).to_frame()
        # alls.mean(1).to_frame()
        allsslopes.loc[:, "comparison"] = "slope"
        allsslopes.loc[:, "freq"] = freq
        allsslopes.columns = ["sum", "comparison", "freq"]
        res.append(allsslopes)

        # Compute correlation between DCD kernel and ACC kernel
        freqs = v1decoding.columns.values
        for lag in range(-2, 3):
            # Example:
            # high level (motor): [N, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # low level (sensor): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, N]
            # Lag = 1
            highlevel_slice = slice(np.abs(lag), 10)
            lowlevel_slice = slice(0, (10 - np.abs(lag)))
            if lag < 0:
                highlevel_slice, lowlevel_slice = lowlevel_slice, highlevel_slice
            x = np.arange(10)
            # Example
            # high level (motor): [ N, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # low level (sensor): [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, N]
            # Lag = -1
            for i in range(15):
                rescorr.append(
                    {
                        "freq": freq,
                        "V1DCD-AFcorr": np.corrcoef(
                            v1decoding.T.values[i, highlevel_slice],
                            alls.values[i, lowlevel_slice],
                        )[0, 1],
                        "M1DCD-AFcorr": np.corrcoef(
                            m1decoding.T.values[i, highlevel_slice],
                            alls.values[i, lowlevel_slice],
                        )[0, 1],
                        "VODCD-AFcorr": np.corrcoef(
                            intdecoding.T.values[i, highlevel_slice],
                            alls.values[i, lowlevel_slice],
                        )[0, 1],
                        "subject": i + 1,
                        "lag": lag,
                    }
                )
    rescorr = pd.DataFrame(rescorr)
    res = pd.concat(res)
    n = 5
    width_ratios = [1, 0.25, 1, 0.25, 0.5]
    if gs is None:
        gs = matplotlib.gridspec.GridSpec(1, n, width_ratios=width_ratios)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, n, subplot_spec=gs, width_ratios=width_ratios
        )
    ax = plt.subplot(gs[0, 0])
    allv1 = pd.pivot_table(
        data=res.query('comparison=="V1kernel"'),
        index="subject",
        columns="freq",
        values="sum",
    )
    plotwerr(allv1, color=figure_6colors["AFcorr"])

    cifrem = pd.pivot_table(
        data=res.query('comparison=="V1kernelCIF"'),
        index="subject",
        columns="freq",
        values="sum",
    )
    plotwerr(cifrem, color=figure_6colors["AF-CIFcorr"])
    # draw_sig(ax,diff, fdr=False, color='k')
    draw_sig(ax, allv1 - cifrem, conjunction=allv1, fdr=True, color="k")
    p = ttest_1samp((allv1 - cifrem).values, 0)[1]
    p2 = ttest_1samp(allv1.values, 0)[1]
    print("Sum sig. frequencies:", allv1.columns.values[(p < 0.05) & (p2 < 0.05)])
    xlabel("Frequency")
    ylabel("Kernel sum")
    yticks([-0.005, 0, 0.005, 0.01])
    sns.despine(ax=ax)
    ax = plt.subplot(gs[0, 2])
    diff = pd.pivot_table(
        data=res.query('comparison=="slope"'),
        index="subject",
        columns="freq",
        values="sum",
    )

    plotwerr(diff)
    p = ttest_1samp(diff.values, 0)[1]
    # print(diff.values.mean(0), p)
    print("Slope sig. frequencies:", diff.columns.values[p < 0.05])
    draw_sig(ax, diff, fdr=False, color=figure_6colors["AFcorr"])
    # draw_sig(ax,diff, fdr=True, color='g')
    xlabel("Frequency")
    ylabel("Slope of\nV1 kernel")
    yticks([-0.002, 0, 0.002])
    sns.despine(ax=ax)
    ax = plt.subplot(gs[0, -1])
    alpha_M1 = pd.pivot_table(
        rescorr.query("10<=freq<=16"),
        index="subject",
        columns="lag",
        values="M1DCD-AFcorr",
    )
    print(rescorr.query("10<=freq<=16").freq.unique())
    alpha_int1 = pd.pivot_table(
        rescorr.query("10<=freq<=16"),
        index="subject",
        columns="lag",
        values="VODCD-AFcorr",
    )
    alpha_V1 = pd.pivot_table(
        rescorr.query("10<=freq<=16"),
        index="subject",
        columns="lag",
        values="V1DCD-AFcorr",
    )
    plotwerr(alpha_M1)
    print("P-valyes Feedback M1->V1:", ttest_1samp(np.arctanh(alpha_M1), 0))
    draw_sig(ax, np.arctanh(alpha_M1), color=figure_6colors["AFcorr"])
    ylabel("Correlation\nV1 Alpha /  M1 decoding")
    yticks([0, 0.15, 0.3])
    xlabel("Lag")
    xticks([-2, 0, 2], ["-2\nV1 leads", 0, "2\nM1 leads"])

    sns.despine(ax=ax)
    return res, rescorr


def _figure6A(cluster="vfcPrimary", freq="gamma", lowfreq=10, gs=None, label=True):
    import matplotlib
    import seaborn as sns
    import pickle

    fs = {
        "gamma": "/Users/nwilming/u/conf_analysis/results/cort_kernel.results.pickle",
        "alpha": "/Users/nwilming/u/conf_analysis/results/cort_kernel_f0-10.results.pickle",
        "beta": "/Users/nwilming/u/conf_analysis/results/cort_kernel_f13-30.results.pickle",
    }
    try:
        a = pickle.load(open(fs[freq], "rb"))
    except KeyError:
        fname = (
            "/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle"
            % freq
        )
        a = pickle.load(open(fname, "rb"))
    ccs, K, kernels, peaks = a["ccs"], a["K"], a["kernels"], a["v1decoding"]
    clpeaks = get_cl_decoding_peaks()
    v1decoding = clpeaks["vfcPrimary"]
    M1decoding = clpeaks["vfcPrimary"]

    v1dcdslopes = v1decoding.apply(lambda x: linregress(np.arange(10), x)[0])
    print(
        "V1 decoding slopes (mean, p, t):",
        np.around(np.mean(v1dcdslopes), 3),
        ttest_1samp(v1dcdslopes, 0),
    )

    fname = (
        "/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle"
        % lowfreq
    )
    b = pickle.load(open(fname, "rb"))
    low_kernels = b["kernels"]

    colors = figure_6colors

    # plt.figure(figsize=(12.5, 5.5))
    if gs is None:
        figure(figsize=(10.5, 3))
        gs = matplotlib.gridspec.GridSpec(
            1, 8, width_ratios=[1, 0.5, 0.25, 1, 1, 0.25, 0.5, 0.5]
        )
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 8, subplot_spec=gs, width_ratios=[1, 0.75, 0.25, 1, 1, 0.55, 0.5, 0.5]
        )
    # gs = matplotlib.gridspec.GridSpec(1, 4)
    ax = plt.subplot(gs[0, 0])
    kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
    KK = np.stack(kernels.kernel)
    kernels = pd.DataFrame(KK, index=kernels.index).query('cluster=="%s"' % cluster)
    rems = pd.pivot_table(data=kernels.query("rmcif==True"), index="subject")
    alls = pd.pivot_table(data=kernels.query("rmcif==False"), index="subject")

    low_kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
    low_KK = np.stack(low_kernels.kernel)
    low_kernels = pd.DataFrame(low_KK, index=low_kernels.index).query(
        'cluster=="%s"' % cluster
    )
    low_rems = pd.pivot_table(data=low_kernels.query("rmcif==True"), index="subject")
    low_alls = pd.pivot_table(data=low_kernels.query("rmcif==False"), index="subject")

    plotwerr(K, color="k", label="Behavioral kernel", lw=1)
    draw_sig(ax, K, color="k")
    plotwerr(v1decoding.T, label="V1 contrast", color=colors["DCD"], lw=1)
    draw_sig(ax, K, y=0.005, color=colors["DCD"])
    plt.yticks([0, 0.04, 0.08, 0.12])
    plt.xticks(np.arange(10), ["1"] + [""] * 8 + ["10"])
    plt.xlabel("Sample")
    plt.ylabel("AUC / Encoding strength")
    # plt.axhline(0, color='k', lw=1)
    plt.legend(
        frameon=False, ncol=1, loc="center left", bbox_to_anchor=[0.2, 1], fontsize=8
    )
    sns.despine(ax=ax, bottom=False, right=True)

    ax = plt.subplot(gs[0, -5], zorder=1)
    plotwerr(alls, label="V1 kernel", color=colors["AFcorr"])
    plotwerr(rems, label="Contrast fluctu-\nations removed", color=colors["AF-CIFcorr"])
    draw_sig(ax, alls, y=-0.01, color=colors["AFcorr"])
    draw_sig(ax, rems, y=-0.01125, color=colors["AF-CIFcorr"])
    draw_sig(ax, alls - rems, y=-0.0125, color="k")
    # plt.xticks(np.arange(10), np.arange(10)+1)
    plt.xticks(np.arange(10), ["1"] + [""] * 8 + ["10"])
    plt.xlabel("Sample")
    plt.ylabel("AUC")
    plt.yticks([0, 0.02, 0.04])
    yl = plt.ylim()
    # plt.axhline(0, color='k', lw=1)
    plt.legend(
        frameon=False, ncol=1, loc="center left", bbox_to_anchor=[0.15, 0.8], fontsize=8
    )
    sns.despine(ax=ax, bottom=False, right=True)
    plt.title("Gamma", fontsize=8)
    ax = plt.subplot(gs[0, -4], zorder=0)

    plotwerr(low_alls, ls=":", color=colors["AFcorr"])  # label='V1 kernel',
    plotwerr(low_rems, ls=":", color=colors["AF-CIFcorr"])
    draw_sig(ax, low_alls, y=-0.01, color=colors["AFcorr"])
    draw_sig(ax, low_rems, y=-0.01125, color=colors["AF-CIFcorr"])
    draw_sig(ax, low_alls - low_rems, y=-0.0125, color="k")

    plt.xticks(np.arange(10), ["1"] + [""] * 8 + ["10"])
    plt.xlabel("Sample")
    plt.ylim(yl)
    plt.ylabel(None)
    plt.yticks([])
    plt.title("Alpha", fontsize=8)
    # plt.axhline(0, color='k', lw=1)

    sns.despine(ax=ax, bottom=False, right=True, left=True)

    rescorr = []
    for i in range(15):
        rescorr.append(
            {
                "AFcorr": np.corrcoef(alls.loc[i + 1], K.loc[i + 1, :])[0, 1],
                "AF-CIFcorr": np.corrcoef(rems.loc[i + 1], K.loc[i + 1, :])[0, 1],
                "V1DCD-AFcorr": np.corrcoef(v1decoding.T.loc[i + 1], K.loc[i + 1, :])[
                    0, 1
                ],
                "M1DCD-AFcorr": np.corrcoef(v1decoding.T.loc[i + 1], K.loc[i + 1, :])[
                    0, 1
                ],
                "DCD": np.corrcoef(v1decoding.T.loc[i + 1], K.loc[i + 1, :])[0, 1],
                "DCD-AFcorr": np.corrcoef(v1decoding.T.loc[i + 1], alls.loc[i + 1, :])[
                    0, 1
                ],
                "lAFcorr": np.corrcoef(low_alls.loc[i + 1], K.loc[i + 1, :])[0, 1],
                "lAF-CIFcorr": np.corrcoef(low_rems.loc[i + 1], K.loc[i + 1, :])[0, 1],
                "subject": i + 1,
            }
        )
    rescorr = pd.DataFrame(rescorr)
    rs = rescorr.set_index("subject").stack().reset_index()
    rs.columns = ["subject", "comparison", "correlation"]
    dcdcorr = rs.query('comparison=="DCD"').correlation.values
    print(
        "DCD correlation with behavior K (mean, p, t):",
        np.around(np.mean(dcdcorr), 2),
        ttest_1samp(dcdcorr, 0),
    )
    ax = plt.subplot(gs[0, -2])
    sns.stripplot(
        data=rs,
        x="comparison",
        y="correlation",
        order=["AFcorr", "AF-CIFcorr"],
        palette=colors,
    )

    plt.ylabel("Correlation with\nbehavioral kernel")
    plt.plot(
        [-0.125, +0.125],
        [rescorr.loc[:, "AFcorr"].mean(), rescorr.loc[:, "AFcorr"].mean()],
        "k",
        zorder=100,
    )
    p = ttest_1samp(np.arctanh(rescorr.loc[:, "AFcorr"]), 0)
    print(
        "M/T/P corr gamma kernel w/ behavior kernel:",
        np.around(np.mean(rescorr.loc[:, "AFcorr"]), 2),
        p,
    )
    if p[1] < 0.05:
        plt.plot(
            [-0.125, +0.125], [-0.95, -0.95], color=colors["AFcorr"], zorder=100, lw=2
        )

    plt.plot(
        [1 - 0.125, 1 + 0.125],
        [rescorr.loc[:, "AF-CIFcorr"].mean(), rescorr.loc[:, "AF-CIFcorr"].mean()],
        "k",
        zorder=100,
    )
    p = ttest_1samp(np.arctanh(rescorr.loc[:, "AF-CIFcorr"]), 0)
    print(
        "M/T/P corr gamma kernel -CIF w/ behavior kernel:",
        np.around(np.mean(rescorr.loc[:, "AF-CIFcorr"]), 2),
        p,
    )
    if p[1] < 0.05:
        plt.plot(
            [1 - 0.125, 1 + 0.125],
            [-0.95, -0.95],
            color=colors["AF-CIFcorr"],
            zorder=100,
            lw=2,
        )
    p = ttest_rel(
        np.arctanh(rescorr.loc[:, "AF-CIFcorr"]), np.arctanh(rescorr.loc[:, "AFcorr"])
    )
    print(
        "M/T/P corr gamma kernel -CIF w/bK vs corr gamma kernel w/bK:",
        np.around(np.mean(rescorr.loc[:, "AF-CIFcorr"] - rescorr.loc[:, "AFcorr"]), 2),
        p,
    )
    plt.title("Gamma", fontsize=8)
    plt.xlabel("")
    plt.xticks([])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.ylim([-1, 1])
    yl = plt.ylim()
    sns.despine(ax=ax, bottom=True)

    ax = plt.subplot(gs[0, -1])
    sns.stripplot(
        data=rs,
        x="comparison",
        y="correlation",
        order=["lAFcorr", "lAF-CIFcorr"],
        palette=colors,
    )
    plt.ylim(yl)
    plt.plot(
        [-0.125, 0 + 0.125],
        [rescorr.loc[:, "lAFcorr"].mean(), rescorr.loc[:, "lAFcorr"].mean()],
        "k",
        zorder=100,
    )
    p = ttest_1samp(np.arctanh(rescorr.loc[:, "lAFcorr"]), 0)
    print(
        "M/T/P corr alpha kernel -CIF w/ behavior kernel:",
        np.around(np.mean(rescorr.loc[:, "lAFcorr"]), 2),
        p,
    )
    if p[1] < 0.05:
        plt.plot(
            [-0.125, +0.125], [-0.95, -0.95], color=colors["AFcorr"], zorder=100, lw=2
        )
    plt.plot(
        [1 - 0.125, 1 + 0.125],
        [rescorr.loc[:, "lAF-CIFcorr"].mean(), rescorr.loc[:, "lAF-CIFcorr"].mean()],
        "k",
        zorder=100,
    )
    p = ttest_1samp(np.arctanh(rescorr.loc[:, "lAF-CIFcorr"]), 0)
    print(
        "M/T/P corr alpha kernel -CIF w/ behavior kernel:",
        np.around(np.mean(rescorr.loc[:, "AF-CIFcorr"]), 2),
        p,
    )
    if p[1] < 0.05:
        plt.plot(
            [1 - 0.125, 1 + 0.125],
            [-0.95, -0.95],
            color=colors["AF-CIFcorr"],
            zorder=100,
            lw=2,
        )
    p = ttest_rel(
        np.arctanh(rescorr.loc[:, "lAF-CIFcorr"]), np.arctanh(rescorr.loc[:, "lAFcorr"])
    )
    print(
        "M/T/P corr alpha kernel -CIF w/bK vs corr alpha kernel w/bK:",
        np.around(np.mean(rescorr.loc[:, "AF-CIFcorr"] - rescorr.loc[:, "lAFcorr"]), 2),
        p,
    )
    plt.title("Alpha", fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("")
    plt.xlabel("")
    if label:
        # plt.xticks([0, 1, 2], ['V1 kernel', 'V1 contrast\ndecoding', 'Contrast\nfluctations\nremoved'], rotation=45)
        plt.xticks([])

    else:
        plt.xticks([0, 1, 2], ["", "", ""], rotation=45)
    # plt.plot([3-0.125, 3+0.125], [rescorr.loc[:, 'DCD-AFcorr'].mean(), rescorr.loc[:, 'DCD-AFcorr'].mean()], 'k')
    # plt.xlim([-0.5, 2.5])
    plt.xlabel("")
    sns.despine(ax=ax, bottom=True, left=True)

    ax = plt.subplot(gs[0, 1], zorder=-1)
    pal = {}
    for cluster in peaks.columns.get_level_values("cluster").unique():
        r = []
        for i in range(15):
            r.append(np.corrcoef(peaks[cluster].T.loc[i + 1], K.loc[i + 1, :])[0, 1])
        pal[cluster] = np.mean(r)

    # pal = dict(ccs.groupby('cluster').mean().loc[:, 'AFcorr'])
    img = _get_lbl_annot_img(
        {k: v + 1 for k, v in pal.items()}, low=1 + -1, high=1 + 1, thresh=-1
    )
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)
    return rescorr


def plot_per_sample_resp(ax, ssd, area, area_label, integration_slice, ylim=None):
    k = dp.sps_2lineartime(dp.get_ssd_per_sample(ssd, "SSD", area=area))
    ka = dp.sps_2lineartime(dp.get_ssd_per_sample(ssd, "SSD_acc_contrast", area=area))

    samples_k = {i: [] for i in range(10)}
    samples_ka = {i: [] for i in range(10)}
    # Compute significances
    for subject in range(1, 16):
        _k = dp.sps_2lineartime(
            dp.get_ssd_per_sample(ssd.query("subject==%i" % subject), "SSD", area=area)
        )
        _ka = dp.sps_2lineartime(
            dp.get_ssd_per_sample(
                ssd.query("subject==%i" % subject), "SSD_acc_contrast", area=area
            )
        )
        for sample in range(10):
            i = _k.loc[:, sample].to_frame()
            i.loc[:, "subject"] = subject
            i = i.set_index("subject", append=True)
            samples_k[sample].append(i)
            i = _ka.loc[:, sample].to_frame()
            i.loc[:, "subject"] = subject
            i = i.set_index("subject", append=True)
            samples_ka[sample].append(i)

    samples_k = {
        k: pd.pivot_table(
            pd.concat(v).dropna(), index="subject", columns="latency", values=k
        )
        for k, v in samples_k.items()
    }
    samples_ka = {
        k: pd.pivot_table(
            pd.concat(v).dropna(), index="subject", columns="latency", values=k
        )
        for k, v in samples_ka.items()
    }

    for sample in range(1, 10):
        # print(area, (samples_k[sample]-samples_ka[sample]).loc[:, 0.1*sample:].head())
        draw_sig(
            ax,
            (samples_k[sample] - samples_ka[sample]).loc[:, 0.1 * sample :],
            fdr=True,
            color="k",
        )

    plt.plot(ka.index, ka.max(1), "m", label="Accumulated\ncontrast")
    plt.plot(ka.index, ka, "m", lw=0.1)

    plt.plot(k.index, k.max(1), color=figure_6colors["DCD"], label="Contrast")
    plt.plot(k.index, k, color=figure_6colors["DCD"], lw=0.1)

    plt.ylim(ylim)
    plt.legend()
    yl = plt.ylim()
    plt.fill_between(
        [integration_slice.start, integration_slice.stop],
        [yl[0], yl[0]],
        [yl[1], yl[1]],
        facecolor="k",
        alpha=0.125,
        zorder=-1,
        edgecolor="none",
    )
    plt.title(area_label)
    plt.xlabel("Time")
    plt.ylabel("Encoding strength")


@memory.cache()
def _get_palette(palette, brain=None, ogl=False, views=["par", "med"]):
    brain = dp.plot_brain_color_legend(
        palette, brain=brain, ogl=ogl, subject="fsaverage"
    )
    return brain.save_montage("/Users/nwilming/Desktop/t.png", views)


@memory.cache()
def _get_lbl_annot_img(
    palette, low=0.4, high=0.6, views=[["lat"], ["med"]], colormap="RdBu_r", thresh=0
):
    print(colormap)
    brain, non_itms = dp.plot_brain_color_annotations(
        palette, low=low, high=high, alpha=1, colormap=colormap
    )
    if len(non_itms) > 0:
        import matplotlib

        norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
        cm = matplotlib.cm.get_cmap(colormap)
        non_itms = {key: np.array(cm(norm(val))) for key, val in non_itms.items()}
        # palette[name] = cm(norm(value))
        brain = dp.plot_brain_color_legend(non_itms, brain=brain, subject="fsaverage")
    return brain.save_montage("/Users/nwilming/Desktop/t.png", views)


@memory.cache()
def _get_img(palette, low=0.3, high=0.7, views=[["lat"], ["med"]]):
    brain, _ = dp.plot_brain_color_annotations(palette, low=low, high=high)
    return brain.save_montage("/Users/nwilming/Desktop/t.png", views)


@memory.cache
def get_cl_decoding_peaks():
    ssd = dp.get_ssd_data(ogl=False, restrict=False)
    p = dp.extract_latency_peak_slope(ssd).test_slope.Pair
    lat = p.vfcPrimary.groupby("subject").mean().mean()
    return dp.extract_peak_slope(ssd, latency=lat).test_slope.Pair


@memory.cache
def get_ogl_decoding_peaks():
    ssd = dp.get_ssd_data(ogl=True, restrict=False)
    p = dp.extract_latency_peak_slope(ssd).test_slope.Pair
    lat = p.V1.groupby("subject").mean().mean()
    return dp.extract_peak_slope(ssd, latency=lat).test_slope.Pair


def make_stimulus(contrast, baseline=0.5, ringwidth=3 / 4):
    contrast = contrast / 2
    low = baseline - contrast
    high = baseline + contrast
    shift = 1
    ppd = 45
    sigma = 75
    cutoff = 5.5
    radius = 4 * ppd
    ringwidth = ppd * ringwidth
    inner_annulus = 1.5 * ppd
    X, Y = np.meshgrid(np.arange(1980), np.arange(1024))

    # /* Compute euclidean distance to center of our ring stim: */
    # float d = distance(pos, RC);
    d = ((X - 1980 / 2) ** 2 + (Y - 1024 / 2) ** 2) ** 0.5

    # /* If distance greater than maximum radius, discard this pixel: */
    # if (d > Radius + Cutoff * Sigma) discard;
    # if (d < Radius - Cutoff * Sigma) discard;
    # if (d < Annulus) discard;

    # float alpha = exp(-pow(d-Radius,2.)/pow(2.*Sigma,2.));
    alpha = np.exp(-(d - radius) ** 2 / (2 * sigma) ** 2)

    # /* Convert distance from units of pixels into units of ringwidths, apply shift offset: */
    # d = 0.5 * (1.0 + sin((d - Shift) / RingWidth * twopi));

    rws = 0.5 * (1.0 + np.sin((d - shift) / ringwidth * 2 * np.pi))

    # /* Mix the two colors stored in gl_Color and secondColor, using the slow
    # * sine-wave weight term in d as a mix weight between 0.0 and 1.0:
    # */
    # gl_FragColor = ((mix(firstColor, secondColor, d)-0.5) * alpha) + 0.5;
    rws = high * (1 - rws) + low * rws
    rws[d > (radius + cutoff * sigma)] = 0.5
    rws[d < (radius - cutoff * sigma)] = 0.5
    rws[d < inner_annulus] = 0.5

    return rws


def get_buildup_slopes(df, tmin=-0.25, dt=0.25):
    X = df.query(
        "epoch=='stimulus' & cluster=='JWG_M1' & contrast=='choice' & ~(hemi=='avg')"
    )

    times = X.columns.values
    times = times[tmin < times]
    # print(times)
    res = []
    for t in times:
        slopes = _get_slopes(X, [t, t + dt])
        k = [
            {"subject": i + 1, "time": t + dt, "dt": dt, "slope": s}
            for i, s in enumerate(slopes)
        ]
        res.extend(k)
    return pd.DataFrame(res)


def get_slopes(df, time):
    X = df.query(
        "epoch=='stimulus' & cluster=='JWG_M1' & contrast=='choice' & ~(hemi=='avg')"
    )
    return _get_slopes(X, time)


def _get_slopes(X, time):

    slopes = []
    inters = []
    for subject in range(1, 16):
        T = (
            pd.pivot_table(
                data=X.query("subject==%i & 10<freq & freq<40" % subject), index="freq"
            )
            .loc[:, time[0] : time[1]]
            .mean(0)
        )
        x, y = T.index.values, T.values
        s, i, _, _, _ = linregress(x, y)
        slopes.append(s)
        inters.append(i)
        # plot(x, y)
        # plot(x, x*s+i)

    return np.array(slopes)


def get_decoding_buildup(ssd, area="JWG_M1"):
    k = []
    for subject in range(1, 16):
        _ka = (
            dp.sps_2lineartime(
                dp.get_ssd_per_sample(
                    ssd.test_slope.Lateralized.query("subject==%i" % subject),
                    "SSD_acc_contrast",
                    area=area,
                )
            )
            .mean(1)
            .to_frame()
        )
        _ka.loc[:, "subject"] = subject
        k.append(_ka)
    k = pd.concat(k)
    return pd.pivot_table(k, index="subject", columns="latency", values=0)


def _decoding_buildup_slopes(X, dt=0.1):
    times = X.columns.values
    dtime = np.diff(times)[0]
    S = []
    for t in times[(times > (times.min() + dt)) & (times < (times.max() - dt))]:

        s = (
            X.loc[:, t - dt : t + dt]
            .T.apply(lambda x: linregress(np.arange(len(x)) * dtime, x)[0])
            .to_frame()
        )

        s.loc[:, "latency"] = t
        s.columns = ["slope", "latency"]
        S.append(s)
    S = pd.concat(S)
    return pd.pivot_table(S, index="subject", columns="latency", values="slope")


def _cp_corr():
    import pickle

    rescorr = []
    pairwise = []
    for freq in list(range(1, 10)) + list(range(10, 115, 5)):
        fname = (
            "/Users/nwilming/u/conf_analysis/results/ncort_kernel_f%i.results.pickle"
            % freq
        )
        a = pickle.load(open(fname, "rb"))
        ccs, K, kernels_d, _ = a["ccs"], a["K"], a["kernels"], a["v1decoding"]
        kernels_d.set_index(["cluster", "rmcif", "subject"], inplace=True)
        KK = np.stack(kernels_d.kernel)
        for cluster in [
            "vfcPrimary",
            "vfcEarly",
            "vfcV3ab",
            "vfcIPS01",
            "vfcIPS23",
            "JWG_aIPS",
            "JWG_IPS_PCeS",
            "JWG_M1",
        ]:
            kernels = pd.DataFrame(KK, index=kernels_d.index).query(
                'cluster=="%s"' % cluster
            )
            alls = pd.pivot_table(data=kernels.query("rmcif==False"), index="subject")
            for i in range(15):
                rescorr.append(
                    {
                        "AFcorr": np.corrcoef(alls.loc[i + 1], K.loc[i + 1, :])[0, 1],
                        "sum": (alls.loc[i + 1] ** 2).sum(),
                        "slope": linregress(np.arange(10), alls.loc[i + 1])[0],
                        "subject": i + 1,
                        "freq": freq,
                        "cluster": cluster,
                    }
                )
            for cluster2 in [
                "vfcPrimary",
                "vfcEarly",
                "vfcV3ab",
                "vfcIPS01",
                "vfcIPS23",
                "JWG_aIPS",
                "JWG_IPS_PCeS",
                "JWG_M1",
            ]:
                kernels2 = pd.DataFrame(KK, index=kernels_d.index).query(
                    'cluster=="%s"' % cluster2
                )
                alls2 = pd.pivot_table(
                    data=kernels2.query("rmcif==False"), index="subject"
                )
                for i in range(15):
                    pairwise.append(
                        {
                            "corr": np.corrcoef(alls.loc[i + 1], alls2.loc[i + 1])[
                                0, 1
                            ],
                            "freq": freq,
                            "c1": cluster,
                            "c2": cluster2,
                        }
                    )
    rescorr = pd.DataFrame(rescorr)
    pairwise = pd.DataFrame(pairwise)
    return rescorr.set_index(["subject", "cluster", "freq"]), pairwise


def _supp_leakage(ccs, pairs):
    plt.figure(figsize=(10, 15))
    import pickle

    areas = [
        "vfcPrimary",
        "vfcEarly",
        "vfcV3ab",
        "vfcIPS01",
        "vfcIPS23",
        "JWG_aIPS",
        "JWG_IPS_PCeS",
        "JWG_M1",
    ]
    k10 = pickle.load(
        open(
            "/Users/nwilming/u/conf_analysis/results/ncort_kernel_f10.results.pickle",
            "rb",
        )
    )
    kernels10 = k10["kernels"]
    k55 = pickle.load(
        open(
            "/Users/nwilming/u/conf_analysis/results/ncort_kernel_f55.results.pickle",
            "rb",
        )
    )
    kernels55 = k55["kernels"]
    kernels10.set_index(["cluster", "rmcif", "subject"], inplace=True)
    kernels55.set_index(["cluster", "rmcif", "subject"], inplace=True)

    gs = matplotlib.gridspec.GridSpec(3, 1, height_ratios=[1.2, 1, 0.8])
    gs_a = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[0, 0])
    for i, cluster in enumerate(areas):
        KK10 = np.stack(kernels10.kernel)
        kk10 = pd.DataFrame(KK10, index=kernels10.index).query(
            'cluster=="%s"' % cluster
        )
        alls10 = pd.pivot_table(data=kk10.query("rmcif==False"), index="subject")
        KK55 = np.stack(kernels55.kernel)
        kk55 = pd.DataFrame(KK55, index=kernels55.index).query(
            'cluster=="%s"' % cluster
        )
        alls55 = pd.pivot_table(data=kk55.query("rmcif==False"), index="subject")
        plt.subplot(gs_a[i // 3, np.mod(i, 3)])
        plotwerr(alls10, label="10Hz")
        plotwerr(alls55, label="55Hz")
        plt.title(cluster)
        plt.ylim([-0.015, 0.035])
        if i > 0:
            plt.yticks([-0.01, 0, 0.01, 0.02, 0.03], [])
        else:
            plt.yticks([-0.01, 0, 0.01, 0.02, 0.03], [-0.01, 0, 0.01, 0.02, 0.03])
    plt.legend()
    sns.despine()

    CC = (
        ccs.query("(9<freq<16) | (50<=freq<=55)")
        .groupby(
            [
                "subject",
                "cluster",
                pd.cut(
                    ccs.query("(9<freq<16) | (50<=freq<=55)").reset_index().freq.values,
                    [0, 9, 16, 49, 59, 110],
                ),
            ]
        )
        .mean()
        .reset_index()
        .dropna()
    )
    CC.loc[:, "freq"] = [c.mid for c in CC.level_2]
    del CC["level_2"]
    CC.set_index(["freq", "subject", "cluster"], inplace=True)
    gs_b = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2, 0])
    CCC = CC.stack().reset_index()
    CCC.columns = ["freq", "subject", "cluster", "measure", "values"]
    for i, measure in enumerate(["sum", "slope", "AFcorr"]):
        g = sns.pointplot(
            data=CCC.groupby(["freq", "cluster", "measure"])
            .mean()
            .query('measure=="%s"' % measure)
            .reset_index(),
            x="cluster",
            y="values",
            height=3.5,
            aspect=1.2,
            sharey=False,
            col="measure",
            hue="freq",
            ax=subplot(gs_b[0, i]),
            order=areas,
        )
        plt.title(measure)
        g.set_xticklabels(areas, rotation=30)
        # g.add_legend()
    gs_c = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0])
    ax = plt.subplot(gs_c[0, 0])
    X55 = (
        pd.pivot_table(
            data=pairs.query("freq==50"), index="c1", columns="c2", values="corr"
        )
        + pd.pivot_table(
            data=pairs.query("freq==55"), index="c1", columns="c2", values="corr"
        )
    ) / 2
    X55 = X55.loc[areas, areas]
    plt.imshow(X55, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(X55)))
    ax.set_xticklabels(X55.columns, rotation=90)
    ax.set_yticks(np.arange(len(X55)))
    ax.set_yticklabels(X55.index)
    plt.title("[50-55]HZ")
    ax = plt.subplot(gs_c[0, 1])
    X55 = (
        pd.pivot_table(
            data=pairs.query("freq==10"), index="c1", columns="c2", values="corr"
        )
        + pd.pivot_table(
            data=pairs.query("freq==15"), index="c1", columns="c2", values="corr"
        )
    ) / 2
    X55 = X55.loc[areas, areas]
    plt.imshow(X55, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.title("[10-15]HZ")
    ax.set_xticks(np.arange(len(X55)))
    ax.set_xticklabels(X55.columns, rotation=90)
    plt.colorbar()
    # ax.set_yticks(np.arange(len(X55)))
    # ax.set_yticklabels(X55.index)
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_leakage.pdf"
    )


def print_timings(timings=None):
    if timings is None:
        timings = pd.read_hdf("/Users/nwilming/u/conf_analysis/results/all_timing.hdf")
    for sub, t in timings.groupby("snum"):
        RD = (t.ref_offset_t - t.ref_onset_t) / 1200
        pp = "Ref: [{:1.2f}-{:1.2f}], {:1.2f}".format(RD.min(), RD.max(), RD.mean())
        ref_delay = (t.stim_onset_t - t.ref_offset_t) / 1200
        pp += "R-S: [{:1.2f}-{:1.2f}], {:1.2f}".format(
            ref_delay.min(), ref_delay.max(), ref_delay.mean()
        )
        RT = (t.button_t - t.stim_offset_t) / 1200
        pp += " || S-RT: [{:1.2f}-{:1.2f}], {:1.2f}".format(
            RT.min(), RT.max(), RT.mean()
        )
        FB = (t.meg_feedback_t - t.button_t) / 1200
        pp += " || RT-FB: [{:1.2f}-{:1.2f}], {:1.2f}; [{:1.2f}, {:1.2f}, {:1.2f}, {:1.2f}]".format(
            FB.min(),
            FB.max(),
            FB.mean(),
            *np.percentile(FB.dropna(), [25, 50, 75, 100])
        )
        delay = []
        for (d, b), tt in t.groupby(["day", "block_num"]):
            delay.append(
                (
                    (
                        tt.ref_onset_t.iloc[10:-10].values
                        - tt.meg_feedback_t.iloc[9:-11].values
                    )
                    / 1200
                )
            )
        delay = np.concatenate(delay)

        pp += " || delay: [{:1.2f}-{:1.2f}], {:1.2f}".format(
            np.nanmin(delay), np.nanmax(delay), np.nanmean(delay)
        )
        print("{:>2s}".format(str(sub)), pp)

    return timings
