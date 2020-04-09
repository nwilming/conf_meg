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

import matplotlib.font_manager as font_manager
path = '/Users/nwilming/font_copies/Helvetica-01.ttf'
prop = font_manager.FontProperties(fname=path)
# Set font property dict
matplotlib.rcParams['font.family'] = 'Helvetica'

rc = {"font.size": 7.0, 'xtick.labelsize':7.0, 
    #'xlabel.font.size':7.0, 
    #'ylabel.font.size':7.0, 
    'ytick.labelsize':7.0,
    'legend.fontsize':7.0,
    "lines.linewidth": 1, 
    'font.family':prop.get_name()}
#label_size = 8
#mpl.rcParams['xtick.labelsize'] = label_size 

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
    ax, pivottable, y=0, color="k", fdr=False, lw=2, p=0.05, conjunction=None, **kwargs
):
    from scipy.stats import ttest_1samp

    p_sig = ttest_1samp(pivottable, 0)[1]
    if fdr:
        from mne.stats import fdr_correction

        id_sig, _ = fdr_correction(p_sig)
        id_sig = list(id_sig)
    else:
        id_sig = list(p_sig < p)

    if conjunction is not None:
        p_con_sig = ttest_1samp(conjunction, 0)[1]
        id_con_sig = p_con_sig < p
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

    with mpl.rc_context(rc=rc):
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
            #(4, "Delay"),
            (5, "Contrast"),
            (4, "Response (choice/confidence)"),
            #(2, "Feedback delay"),
            (3, "Audio feedback"),
        ]
        time = np.linspace(-1.25, 2.1, 5000)
        ref_time = np.linspace(-1.25, -0.1, 5000)
        stim_time = np.linspace(-0.1, 2.1, 5000)
        # Reference
        ax.plot(ref_time, 5 + set(-0.8, -0.4, 0.5, x=ref_time), "k", zorder=5)
        ax.text(-0.35, 5.35, "0.4s", va="center")
        # Reference delay
        #ax.plot(time, 4 + set(-0.4, -0.0, 0.5, set_nan=True), ":k", zorder=5)
        # ax.plot(time, 4 + set(-100, -200.0, 0.5, set_nan=False), "k", zorder=5)
        #ax.plot([time.min(), -0.4], [4, 4], "k", zorder=5)
        #ax.plot([0, time.max()], [4, 4], "k", zorder=5)
        #ax.text(0.05, 4.25, "1-1.5s", va="center")
        # Test stimulus
        cvals = array([0.71, 0.33, 0.53, 0.75, 0.59, 0.57, 0.55, 0.61, 0.45, 0.58])
        #colors = sns.color_palette(n_colors=10)
        
        norm=matplotlib.colors.Normalize(-5, 10)
        cm = matplotlib.cm.get_cmap('BuPu')
        colors = [cm(norm(10-i)) for i, c in enumerate(cvals)]
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
        ax.text(1.05, 5.35, "100ms/sample", va="center")
        # Response
        ax.plot(time, 4 + set(1.449, 1.45, 0.5, set_nan=True), "k", zorder=5)
        ax.plot([time.min(), time.max()], [4, 4], "k", zorder=5)
        #ax.plot([1.45, time.max()], [3, 3], "k", zorder=5)
        #Z2 = plt.imread('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/resp.png')
        #aspect = Z2.shape[1]/Z2.shape[0]
        #print('IMG Aspect:', aspect)
        #tt = 0.13
        #axins = ax.inset_axes([0.85, 0.42, tt*aspect, tt], zorder=-1)
        
        #axins.imshow(np.flipud(Z2), extent=[aspect,0, 1, 0], interpolation="nearest",
        #  origin="lower")
        #axins.set_xticks([])
        #axins.set_yticks([])
        #sns.despine(ax=axins, left=True, bottom=True)
        #ax.text(1.5, 4.25, "0.45s avg", va="center")
        # Feedback delay
        #ax.plot(time, 2 + set(1.45, 1.65, 0.5), ":k", zorder=5)
        # ax.plot(time, 2 + set(100.35, 100.55, 0.5), "k", zorder=5)
        #ax.text(1.7, 2.25, "0-1.5s", va="center")
        #ax.plot([time.min(), 1.45], [2, 2], "k", zorder=5)
        #ax.plot([1.65, time.max()], [2, 2], "k", zorder=5)
        # Feedback
        ax.plot(time, 3 + set(1.65, 1.65 + 0.25, 0.5), "k", zorder=5)
        ax.text(1.925, 3.35, "0.25s", va="center")

        ax.set_yticks([])  # i[0] for i in yticks])
        # ax.set_yticklabels([i[1] for i in yticks], va='bottom')
        for y, t in yticks:
            ax.text(-1.25, y + 0.35, t, verticalalignment="center")
        
        ax.set_xticks([])
        ax.tick_params(axis=u"both", which=u"both", length=0)
        ax.set_xlim(-1.25, 2.1)
        ax.set_ylim(2.8, 6.6)
        sns.despine(ax=ax, left=True, bottom=True)
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


def invax(fig, gs):
    ax = fig.add_subplot(gs, zorder=-10)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)
    return ax


def add_letter(fig, gs, label, x=-0.05, y=1.1):
    ax = invax(fig, gs)
    ax.text(x, y, label, transform=ax.transAxes,
      fontsize=10, fontweight='bold', va='top', ha='right')
    return ax


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


def figure1(data=None, slow=False):
    from conf_analysis.behavior import empirical, kernels
    from conf_analysis import behavior

    color_palette = behavior.parse(behavior.colors)
    if data is None:
        data = empirical.load_data()
    data.loc[:, "choice"] = (data.response + 1) / 2
    data.loc[:, "pconf"] = data.confidence - 1

    fig=figure(figsize=(5.5, 5))
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 0.6], hspace=0.15)

    figure0(gs[0, 0])
    add_letter(fig, gs[0,0], 'A', x=-0.071, y=1.1)
    #sns.despine(ax=plt.gca(), left=True, bottom=True)
    
    with mpl.rc_context(rc=rc):
        #gs_new = matplotlib.gridspec.GridSpecFromSubplotSpec(
        #    2, 6, gs[1, 0], wspace=1.5, hspace=0.15
        #)
        # gs = matplotlib.gridspec.GridSpec(2, 6, wspace=1.5, hspace=0.5)

        
        
        """
        kernels.get_decision_fluct(data)
        add_letter(fig, gs[0,:2], 'B', x=-0.15)

        ax = subplot(gs[0, 2:])
        from conf_analysis.behavior import kernels
        

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
        """

        #print(
        #    "Slopes for confidence kernel (error: mean, t, p, #<0):",
        #    np.mean(serr),
        #    ttest_1samp(serr, 0),
        #    sum(np.array(serr) < 0),
        #)

        #---> Comment in for kernel:
        from conf_analysis.meg import cort_kernel as ck
        dk, confidence_kernel = ck.get_kernels()
        ax = subplot(gs[1, :])
        plotwerr(dk, color="k", label="Decision kernel", lw=2)
        draw_sig(ax, dk, fdr=True, color="k", y=0.0)
        
        ylim([0.49 - 0.5, 0.64 - 0.5])
        ylabel("AUC-0.5", fontsize=7)
        xlabel("Contrast sample number", fontsize=7)
        legend('', frameon=False)
        yticks(np.array([0.5, 0.55, 0.6]) - 0.5)
        xticks(np.arange(10), np.arange(10) + 1)
        for p in np.arange(10):
            ax.axvline(p, lw=0.5, color='gray', alpha=0.5, zorder=-1000)
        xlim([-.5, 9.5])
        sns.despine(ax=gca(), bottom=False)

        add_letter(fig, gs[1,:], 'B', x=-0.071, y=1.1)
        #kernels.plot_decision_kernel_values(data)
        #ylabel("Contrast")
        #xlabel("Contrast sample")
        #legend(frameon=False)
        #add_letter(fig, gs[0,2:], 'C', x=-0.125)

        #savefig(
        #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/1_figure_1.pdf",
        #    bbox_inches="tight",
        #    dpi=1200,
        #)
        if slow:
            p = data.groupby(["snum"]).apply(
                lambda x: empirical.fit_choice_logistic(x, summary=False)
            )
            p = p.groupby("snum").mean()
            print(
                "Can predict choice above chance in %i/15 subjects. \nMean choice accuracy is %0.2f (%0.2f-%0.2f)"
                % ((sum(p > 0.5)), np.around(np.mean(p), 2), np.around(p.min(), 2), np.around(p.max(), 2))
            )

            p = data.groupby(["snum", "response"]).apply(
                lambda x: empirical.fit_conf_logistic(x, summary=False)
            )
            p = p.groupby("snum").mean()
            print(
                "Can predict confidence above chance in %i/15 subjects. \nMean confidence accuracy is %0.2f (%0.2f-%0.2f)"
                % ((sum(p > 0.5)), np.around(np.mean(p), 2), np.around(p.min(), 2), np.around(p.max(), 2))
            )
    return


def figureS1():
    from conf_analysis.behavior import empirical
    from conf_analysis import behavior
    data = empirical.load_data()
    data.loc[:, "choice"] = (data.response + 1) / 2
    data.loc[:, "pconf"] = data.confidence - 1
    color_palette = behavior.parse(behavior.colors)
    fig=figure(figsize=(7.5, 4.5))
    with mpl.rc_context(rc=rc):
        gs = matplotlib.gridspec.GridSpec(
            2, 6, wspace=1.5, hspace=0.55
        )
            # This panel needs to go into supplements
        subplot(gs[0, :2])
        
        X = pd.pivot_table(index="snum", columns="pconf", values="correct", data=data)
        mean = X.mean(0)
        print(X.mean(1))
        mean_all = X.mean().mean()
        for i in X.index.values:
            plot([0, 1], X.loc[i, :], color="k", alpha=0.25, lw=0.5)

        sem = 2 * X.std(0) / (15 ** 0.5)
        semall = 2 * X.mean(1).std() / (15 ** 0.5)
        print(semall)
        plot([-0.4], mean_all, 'o', color='gray')
        plot([-0.4, -0.4], [semall+mean_all, mean_all-semall], color='gray')
        plot([0], mean[0], "o", color=color_palette["Secondary2"][0])
        plot([1], mean[1], "o", color=color_palette["Secondary1"][0])
        plot(
            [0, 0], [sem[0] + mean[0], mean[0] - sem[0]], color_palette["Secondary2"][0]
        )
        plot(
            [1, 1], [sem[1] + mean[1], mean[1] - sem[1]], color_palette["Secondary1"][0]
        )
        xticks([-0.4, 0, 1], ["All\ntrials", r"Low", r"High"])
        ylabel("% Correct", fontsize=7)
        xlabel("Confidence", fontsize=7)
        xlim(-0.6, 1.2)
        from scipy.stats import ttest_rel, ttest_1samp

        print("T-Test for accuracy by confidence:", ttest_rel(X.loc[:, 0], X.loc[:, 1]))
        sns.despine(ax=gca())

        add_letter(plt.gcf(), gs[0, :2], 'A', x=-0.25)
        subplot(gs[0, 4:])

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
        xlabel("Evidence discriminability", fontsize=7)
        ylabel("Confidence", fontsize=7)
        yticks([0.2, 0.3, 0.4, 0.5, 0.6])
        sns.despine(ax=gca())
        add_letter(plt.gcf(), gs[0, 4:], 'C', x=-0.25)

        subplot(gs[0, 2:4])
        dz = (
            data.groupby(["snum", "confidence"])
            .apply(lambda x: by_discrim(x, abs=True))
            .reset_index()
        )
        dz.loc[:, "Evidence discriminability"] = dz.threshold_units
        dall = (
            data.groupby(["snum"])
            .apply(lambda x: by_discrim(x, abs=True))
            .reset_index()
        )
        dall.loc[:, "Evidence discriminability"] = dall.threshold_units
        dall = pd.pivot_table(
            index="snum",
            columns="Evidence discriminability",
            values="accuracy",
            data=dall,
        )        
        plotwerr(
           dall,
            color="gray",
            alpha=1,
            lw=2,
            label="All",
        )

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
        xticks([1, 2], [r"t", r"2t"], fontsize=7)
        xlabel("Evidence discriminability", fontsize=7)
        ylabel("% Correct", fontsize=7)
        yticks([0.5, 0.75, 1], [50, 75, 100], fontsize=7)
        sns.despine(ax=gca())
        # tight_layout()
        add_letter(plt.gcf(), gs[0, 2:4], 'B', x=-0.3)

        ax = subplot(gs[1, :4])
        #Add Panel 1B here!
        # This is the kernel panel
        #ax = subplot(gs[0, :])
        palette = {
            r"$E_{N}^{High}$": '#c64588', # color_palette["Secondary1"][0],
            r"$E_{N}^{Low}$": '#8445c6', #color_palette["Secondary2"][1],
            r"$E_{S}^{Low}$": '#8445c6', #color_palette["Secondary2"][1],
            r"$E_{S}^{High}$": '#c64588', #color_palette["Secondary1"][0],
        }

        
        k = empirical.get_confidence_kernels(data, contrast_mean=0.5)
        for kernel, kdata in k.groupby("Kernel"):
            kk = pd.pivot_table(
                index="snum", columns="time", values="contrast", data=kdata
            )
            if 'E_{N}' in kernel:
                plotwerr(kk, '--', color=palette[kernel], lw=2, label="Low confidence")
            else:
                plotwerr(kk, color=palette[kernel], lw=2, label="Low confidence")
            print(kernel)
        #empirical.plot_kernel(k, palette, legend=False)
        plt.ylabel(r"$\Delta$ Contrast", fontsize=7)
        ax.annotate('Choice: test stronger, high confidence', color=palette[r"$E_{N}^{High}$"], 
            xy=(1.1, 0.03), xytext=(3.5, 0.04),# xycoords='axes', 
            fontsize=7, ha='center', va='bottom',            
            arrowprops=dict(facecolor='black', width=0.5,
                headwidth=3.5, headlength=3.5,
                color=palette[r"$E_{N}^{High}$"]), zorder=10)
        ax.annotate('Choice: test stronger, low confidence', color=palette[r"$E_{N}^{Low}$"], 
            xy=(2.0, 0.0075), xytext=(4.5, 0.025),# xycoords='axes', 
            fontsize=7, ha='center', va='bottom',            
            arrowprops=dict(facecolor='black', width=0.5,
                headwidth=3.5, headlength=3.5,
                color=palette[r"$E_{N}^{Low}$"], zorder=-10), zorder=-10)
        ax.annotate('Choice: test weaker, low confidence', color=palette[r"$E_{N}^{Low}$"], 
            xy=(3.7, -0.009), xytext=(6, -0.025),# xycoords='axes', 
            fontsize=7, ha='center', va='bottom',            
            arrowprops=dict(facecolor='black', width=0.5,
                headwidth=3.5, headlength=3.5,
                color=palette[r"$E_{N}^{Low}$"], zorder=-10), zorder=-10)
        ax.annotate('Choice: test weaker, high confidence', color=palette[r"$E_{N}^{High}$"], 
            xy=(1.1, -0.03), xytext=(3.5, -0.04),# xycoords='axes', 
            fontsize=7, ha='center', va='bottom',            
            arrowprops=dict(facecolor='black', width=0.5,
                headwidth=3.5, headlength=3.5,
                color=palette[r"$E_{N}^{High}$"]), zorder=10)

        for p in np.arange(10):
            ax.axvline(p, lw=0.5, color='gray', alpha=0.5, zorder=-1000)
        #plt.text(
        #    -0.2,
        #    0.003,
        #    "ref.    test",
        #    rotation=90,
        #    horizontalalignment="center",
        #    verticalalignment="center",
        #)
        xlabel("Contrast sample number", fontsize=7)
        # legend(frameon=False)
        xticks(np.arange(10), np.arange(10) + 1, fontsize=7)
        #ax.axhline(color="k", lw=1)
        ax.plot([0, 9], [0, 0], 'k')
        ax.set_ylim(-0.045, 0.045)
        xlim([-.5, 9.5])
        sns.despine(ax=ax)        

        add_letter(plt.gcf(), gs[1, :], 'D', x=-0.1)

        #savefig(
        #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/S1_figure_S1.pdf",
        #    bbox_inches="tight",
        #    dpi=1200,
        #)


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
    with mpl.rc_context(rc=rc):
        figure(figsize=(7.5, 7.5 / 2))
        from conf_analysis.meg import srtfr

        # gs = matplotlib.gridspec.GridSpec(3, 2, width_ratios=[0.99, 0.01])

        fig = srtfr.plot_stream_figures(
            df.query('hemi=="avg"'),
            contrasts=["all"],
            flip_cbar=True,
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


def figure2_alt(df=None, stats=False, dcd=None, aspect='auto'):
    """
    Plot TFRs underneath each other.
    """
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

    if dcd is None:
        dcd = dp.get_decoding_data()
    palette = _stream_palette()

    with mpl.rc_context(rc=rc):
        fig = figure(figsize=(7.5, 7.5))
        from conf_analysis.meg import srtfr

        gs = matplotlib.gridspec.GridSpec(4, 1, height_ratios=[1,1, 1,0.1])

        srtfr.plot_stream_figures(
            df.query('(hemi=="avg")'),
            contrasts=["all"],
            flip_cbar=False,
            gs=gs[0, 0],
            stats=stats,
            aspect=aspect,
            title_palette=palette,
        )
        
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-50, 50), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        cax = fig.add_axes([0.74, 0.685, 0.1, 0.0125 / 2])
        cb = colorbar(
            cmap,
            cax=cax,
            shrink=0.5,
            ticks=[-50, 0, 50],
            drawedges=False,
            orientation="horizontal",            
        )
        cb.set_label(label="% change", fontsize=7)
        
        cb.outline.set_visible(False)
        view_one = dict(azimuth=-40, elevation=100, distance=350)
        view_two = dict(azimuth=-145, elevation=70, distance=350)
        img = _get_palette(palette, views=[view_two, view_one])
        iax = fig.add_axes([0.09, 0.8, 0.19, 0.19])
        iax.imshow(img)
        iax.set_xticks([])
        iax.set_yticks([])                
        lax = add_letter(plt.gcf(), gs[0,0], 'A', x=-0.08, y=1.25)        
        sns.despine(ax=lax, left=True, bottom=True)
        sns.despine(ax=iax, left=True, bottom=True)
        

        srtfr.plot_stream_figures(
            df.query('(hemi=="avg")'),
            contrasts=["stimulus"],
            flip_cbar=False,
            gs=gs[1, 0],
            stats=stats,
            aspect=aspect,
            title_palette=palette,
        )
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-25, 25), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        cax = fig.add_axes([0.74, 0.45, 0.1, 0.0125 / 2])
        cb = colorbar(
            cmap,
            cax=cax,
            shrink=0.5,
            ticks=[-25, 0, 25],
            drawedges=False,
            orientation="horizontal",            
        )
        cb.set_label(label="% change", fontsize=7)
        
        lax = add_letter(plt.gcf(), gs[1,0], 'B', x=-0.08, y=1.1)
        sns.despine(ax=lax, left=True, bottom=True)
        
        
        plotter = dp.StreamPlotter(
            dp.plot_config,
            {"MIDC_split": "k"},#, "CONF_unsigned": "Greens", "CONF_signed": "Blues"},
            {
                "Pair": dcd.test_roc_auc.Pair,
                #"Lateralized": dcd.test_roc_auc.Lateralized
            },
            gs=gs[2, 0],
            title_palette=palette,
        )

        plotter.plot(aspect="auto")

        #cax = fig.add_axes([0.81, 0.27, 0.1, 0.0125 / 2])
        #cax.plot([-10, -1], [0, 0], "r", label="Choice decoding")
        #cax.plot([-10, -1], [0, 0], "b", label="Signed confidence")
        #cax.plot([-10, -1], [0, 0], "g", label="Unigned confidence")
        #cax.set_xlim([0, 1])
        #cax.set_xticks([])
        #cax.set_yticks([])
        #cax.legend(frameon=False)
        #sns.despine(ax=cax, left=True, bottom=True)
        #sns.despine(ax=iax, left=True, bottom=True)
        #sns.despine(ax=lax, left=True, bottom=True)
        add_letter(plt.gcf(), gs[2,0], 'C', x=-0.08, y=1.1)

        ax = subplot(gs[3,0])
        ax.set_xlim([0,1])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.text(0.5, 0.5, 'Association\ncortex', 
            horizontalalignment='center', 
            verticalalignment='center', 
            color=(0.75, 0.75, 0.75))
        plt.text(0.075, 0.5, 'Sensory\ncortex', 
            horizontalalignment='center', 
            verticalalignment='center',
            color=(0.75, 0.75, 0.75))
        plt.text(1-0.075, 0.5, 'Motor\ncortex', 
            horizontalalignment='center', 
            verticalalignment='center',
            color=(0.75, 0.75, 0.75))
        plt.arrow( 0.4, 0.5, -0.2, 0, fc=(0.75, 0.75, 0.75), ec=(0.75, 0.75, 0.75),
             head_width=0.5, head_length=0.01 )
        plt.arrow( 0.6, 0.5, 0.2, 0, fc=(0.75, 0.75, 0.75), ec=(0.75, 0.75, 0.75),
             head_width=0.5, head_length=0.01 )
        sns.despine(ax=ax, left=True, bottom=True)
    # plt.tight_layout()
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/2_figure_2_BothHemisDecoding.pdf",
        dpi=1200
    )
    return df, stats, dcd


def figureS2(df=None, stats=False, aspect="auto"):  # 0.01883834992799947):
    """
    Plot TFRs underneath each other.
    """
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

    #if dcd is None:
    #    dcd = dp.get_decoding_data()
    palette = _stream_palette()

    with mpl.rc_context(rc=rc):
        fig = figure(figsize=(7.5, 10))
        from conf_analysis.meg import srtfr

        gs = matplotlib.gridspec.GridSpec(3, 1)

        srtfr.plot_stream_figures(
            df.query('~(hemi=="avg")'),
            contrasts=["choice"],
            flip_cbar=True,
            gs=gs[1, 0],
            stats=stats,
            aspect=aspect,
            title_palette=palette,
        )

        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-25, 25), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        cax = fig.add_axes([0.74, 0.67, 0.1, 0.0125 / 2])
        cb = colorbar(
            cmap,
            cax=cax,
            shrink=0.5,
            ticks=[-25, 0, 25],
            drawedges=False,
            orientation="horizontal",
            label="% Power change",
        )
        cb.outline.set_visible(False)
        

        srtfr.plot_stream_figures(
            df.query('(hemi=="avg")'),
            contrasts=["stimulus"],
            flip_cbar=False,
            gs=gs[0, 0],
            stats=stats,
            aspect=aspect,
            title_palette=palette,
        )
        add_letter(plt.gcf(), gs[1,0], 'B', x=-0.18, y=1.1)
        add_letter(plt.gcf(), gs[0,0], 'A', x=-0.18, y=1.1)
        """

        plotter = dp.StreamPlotter(
            dp.plot_config,
            {"MIDC_split": "Reds", "CONF_unsigned": "Greens", "CONF_signed": "Blues"},
            {
                # "Averaged": df.test_roc_auc.Averaged,
                "Lateralized": dcd.test_roc_auc.Lateralized
            },
            gs=gs[2, 0],
            title_palette=palette,
        )
        plotter.plot(aspect="auto")

        cax = fig.add_axes([0.81, 0.15, 0.1, 0.0125 / 2])
        cax.plot([-10, -1], [0, 0], "r", label="Choice")
        cax.plot([-10, -1], [0, 0], "b", label="Signed confidence")
        cax.plot([-10, -1], [0, 0], "g", label="Unigned confidence")
        cax.set_xlim([0, 1])
        cax.set_xticks([])
        cax.set_yticks([])
        cax.legend(frameon=False)
        sns.despine(ax=cax, left=True, bottom=True)
        """

    # plt.tight_layout()    
    savefig(
       "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/S2_figure_S2.pdf",
    bbox_inches='tight')
    return df, stats


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

    with mpl.rc_context(rc=rc):
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
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_2B_figure_2B.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    return fig, df, stats


def figureS3(ogldcd=None, pdcd=None):
    if ogldcd is None:
        ogldcd = dp.get_decoding_data(restrict=False, ogl=True)
    if pdcd is None:
        pdcd = pd.read_hdf(
            "/Users/nwilming/u/conf_analysis/results/all_vert_phase_decoding.hdf"
        )
    ogldcd = ogldcd.test_roc_auc.Pair
    ogldcd = ogldcd.query('epoch=="stimulus"')
    plt.figure(figsize=(7, 4))
    gs = matplotlib.gridspec.GridSpec(2, 2, wspace=0.5, hspace=0.4)  # ,height_ratios=[3 / 5, 2 / 5])
    with mpl.rc_context(rc=rc):
        # ----> First OGLDD Line plot!
        ogl_t = 1.083
        signal = "MIDC_split"
        low, high = 0.5, 0.8
        # for i, (signal, data) in enumerate(df.groupby(["signal"])):
        sdata = ogldcd.query('signal=="%s"' % signal)
        ax = plt.subplot(gs[0, 0])
        X = pd.pivot_table(
            index="cluster",
            columns="latency",
            values=0,
            data=sdata.stack().reset_index(),
        )
                
        X = X.loc[:, ::2]
            
        txts = []
        idx = np.argsort(X.loc[:, ogl_t])
        sorting = X.index.values[idx]
        norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
        cm = matplotlib.cm.get_cmap("Reds")
        max_roi = X.loc[:, 1.417].idxmax()

        for j, roi in enumerate(sorting):
            val, color = X.loc[roi, ogl_t], cm(norm(X.loc[roi, ogl_t]))
            if 'V1' == roi:
                print(roi)
                plt.plot(X.columns.values, X.loc[roi, :], color='k', zorder=1000)
            else:
                plt.plot(X.columns.values, X.loc[roi, :], color=color)
            if any([roi == x for x in fig_4_interesting_rois.keys()]) or (
                roi == max_roi
            ):
                try:
                    R = fig_4_interesting_rois[roi]
                except KeyError:
                    R = roi
                if '6d' in roi:
                    txts.append(plt.text(1.417, X.loc[roi, 1.417] + 0.015, R))
                else:
                    txts.append(plt.text(1.417, X.loc[roi, 1.417] - 0.005, R))
        y = np.linspace(0.475, high, 200)
        x = y * 0 - 0.225
        plt.scatter(x, y, c=cm(norm(y)), marker=0)

        plt.title("Spectral power and coarse space\n(hemispheres), whole cortex", fontsize=7)
        plt.xlim([-0.25, 1.4])
        plt.ylim([0.475, high])
        plt.ylabel('AUC', fontsize=7)
        plt.axvline(1.2, color="k", alpha=0.9)
        plt.xlabel("Time", zorder=1, fontsize=7)
        sns.despine(ax=ax)
        add_letter(plt.gcf(), gs[0,0], 'A', x=-0.18, y=1.35)
        # Now plot brain plots.
        palette = {
                d.replace("dlpfc_", "").replace("pgACC_", ""): X.loc[d, ogl_t]
                for d in X.index.values
            }
        # print(palette)
        img = _get_lbl_annot_img(
            palette,
            low=low,
            high=high,
            views=[["par", "front"], ["med", "lat"]],
            colormap='Reds',
        )

        plt.subplot(gs[1, 0], aspect="equal", zorder=-10)
        plt.imshow(img, zorder=-10)
        plt.xticks([])
        plt.yticks([])
        sns.despine(ax=plt.gca(), left=True, bottom=True)
        

        # ---> Now PDCD line plot
        pdcd = pdcd.query("target=='response'")
        pdcd_t = 1.1
        low, high = 0.5, 0.8
        ax = plt.subplot(gs[0, 1])
        X = pd.pivot_table(
            index="roi", columns="latency", values="test_roc_auc", data=pdcd
        )
        txts = []
        idx = np.argsort(X.loc[:, pdcd_t])
        sorting = X.index.values[idx]
        norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
        cm = matplotlib.cm.get_cmap('Reds')
        print(sorting)
        for j, roi in enumerate(sorting):
            val, color = X.loc[roi, pdcd_t], cm(norm(X.loc[roi, pdcd_t]))
            if 'V1' in roi:
                print(roi)
                plt.plot(X.columns.values, X.loc[roi, :], color='k')
            else:
                plt.plot(X.columns.values, X.loc[roi, :], color=color)
            if any([roi == x for x in fig_4B_interesting_rois.keys()]):
                R = fig_4B_interesting_rois[roi]
                txts.append(plt.text(1.4, X.loc[roi, 1.4], R))
        y = np.linspace(0.475, high, 200)
        x = y * 0 - 0.225
        plt.scatter(x, y, c=cm(norm(y)), marker=0)
        plt.title('Spectral power and phase, fine space (vertices)\nand coarse space (hemipheres), selected ROIs', fontsize=7)
        plt.xlim([-0.25, 1.4])
        plt.ylim([0.475, high])
        plt.xlabel("Time", fontsize=7)
        plt.axvline(pdcd_t, color="k", alpha=0.9)
        sns.despine(ax=ax)
        add_letter(plt.gcf(), gs[0,1], 'B', x=-0.18, y=1.35)
        # Now plot brain plots.
        palette = {
                d.replace("dlpfc_", "").replace("pgACC_", ""): X.loc[d, pdcd_t]
                for d in X.index.values
            }
        # print(palette)
        img = _get_lbl_annot_img(
            palette,
            low=low,
            high=high,
            views=[["par", "front"], ["med", "lat"]],
            colormap='Reds',
        )

        ax = plt.subplot(gs[1, 1], aspect="equal", zorder=-10)
        plt.imshow(img, zorder=-10)
        plt.xticks([])
        plt.yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/S3_figure_S3.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    return ogldcd, pdcd


def _dep_figure4(ogldcd=None, pdcd=None):
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
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_3_figure_3.pdf",
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


fig_4_interesting_rois = {"4": "M1", "V1": "V1", "2": "IPS/PostCeS", "7PC": "aIPS"}
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

    with mpl.rc_context(rc=rc):
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
            
            print('Subsetting latencies to remove NANs. Check this if decoding is redone')
            X = X.loc[:, ::2]
            
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
    with mpl.rc_context(rc=rc):
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
    plt.figure(figsize=(8, 11))
    gs = matplotlib.gridspec.GridSpec(
        10, 3, height_ratios=[0.9, 0.9, 0.41, 0.6, 0.25, 0.8, 0.35, 0.5, 0.45, 0.8], hspace=0.1
    )

    with mpl.rc_context(rc=rc):
        # First plot individual lines
        ax = subplot(gs[0, 0])
        plot_per_sample_resp(
            plt.gca(), ssd.test_slope.Averaged, "vfcPrimary", "V1", integration_slice
        )
        legend([], frameon=False)
        xl = plt.xlim()
        norm = matplotlib.colors.Normalize(vmin=-0.05, vmax=0.05)
        cm = matplotlib.cm.get_cmap('RdBu_r')
        y = np.linspace(0, 0.05, 250)
        for i in np.arange(0, 1, 0.1):            
            ax.axvline(i, color='gray', alpha=0.5, zorder=-1000, lw=0.5)
            
        #plt.scatter(-0.15+0*y, y, c=cm(norm(y)), marker=0)
        plt.xlim(xl)
        ax = subplot(gs[0, 1])
        plot_per_sample_resp(
            plt.gca(),
            ssd.test_slope.Lateralized,
            "JWG_IPS_PCeS",
            "IPS/PostCeS",
            integration_slice,
            [-0.005, 0.065],

        )
        plt.ylabel('')
        for i in np.arange(0, 1, 0.1):
            ax.axvline(i, color='gray', alpha=0.5, zorder=-1000, lw=0.5)
        norm = matplotlib.colors.Normalize(vmin=-0.05, vmax=0.05)
        cm = matplotlib.cm.get_cmap('RdBu_r')        
        #plt.scatter(-0.15+0*y, y, c=cm(norm(y)), marker=0)
        plt.xlim(xl)
        ax.set_ylabel("")
        legend([], frameon=False, fontsize=7)
        add_letter(plt.gcf(), gs[0,0], 'A', x=-0.18, y=1.2)
        ax = subplot(gs[0, 2])
        plot_per_sample_resp(
            plt.gca(),
            ssd.test_slope.Lateralized,
            "JWG_M1",
            "M1-hand",
            integration_slice,
            [-0.005, 0.065],
        )
        for i in np.arange(0, 1, 0.1):
            ax.axvline(i, color='gray', alpha=0.5, zorder=-1000, lw=0.5)
        y = np.linspace(0, 0.025, 100)
        norm = matplotlib.colors.Normalize(vmin=-0.025, vmax=0.025)
        cm = matplotlib.cm.get_cmap('RdBu_r')
        #plt.scatter(-0.15+0*y, y, c=cm(norm(y)), marker=0, zorder=1)
        #plt.scatter(-0.115+0*y, y, c=cm(norm(-y)), marker=0, zorder=0)
        plt.xlim(xl)
        ax.set_ylabel("")

        legend('', frameon=False)
        sns.despine()


        # Now plot hull curves
        ax = subplot(gs[1, 0])
        plot_per_sample_resp(
            plt.gca(), ssd.test_slope.Averaged, "vfcPrimary", "", integration_slice,
            hull=True, acc=True, sig=True
        )
        legend([], frameon=False)
        xl = plt.xlim()
        norm = matplotlib.colors.Normalize(vmin=-0.05, vmax=0.05)
        cm = matplotlib.cm.get_cmap('RdBu_r')
        y = np.linspace(0, 0.05, 250)
        for i in np.arange(0, 1, 0.1):            
            ax.axvline(i, color='gray', alpha=0.5, zorder=-1000, lw=0.5)
            
        #plt.scatter(-0.15+0*y, y, c=cm(norm(y)), marker=0)
        plt.xlim(xl)
        ax = subplot(gs[1, 1])
        plot_per_sample_resp(
            plt.gca(),
            ssd.test_slope.Lateralized,
            "JWG_IPS_PCeS",
            "",
            integration_slice,
            [-0.005, 0.065],
            hull=True, acc=True, sig=True,
        )

        for i in np.arange(0, 1, 0.1):
            ax.axvline(i, color='gray', alpha=0.5, zorder=-1000, lw=0.5)
        norm = matplotlib.colors.Normalize(vmin=-0.05, vmax=0.05)
        cm = matplotlib.cm.get_cmap('RdBu_r')        
        #plt.scatter(-0.15+0*y, y, c=cm(norm(y)), marker=0)
        plt.xlim(xl)
        ax.set_ylabel("")
        legend([], frameon=False, fontsize=7)
        add_letter(plt.gcf(), gs[0,0], 'A', x=-0.18, y=1.2)
        ax = subplot(gs[1, 2])
        plot_per_sample_resp(
            plt.gca(),
            ssd.test_slope.Lateralized,
            "JWG_M1",
            "",
            integration_slice,
            [-0.005, 0.065],
            hull=True,
            acc=True, sig=True
        )
        for i in np.arange(0, 1, 0.1):
            ax.axvline(i, color='gray', alpha=0.5, zorder=-1000, lw=0.5)
        y = np.linspace(0, 0.025, 100)
        norm = matplotlib.colors.Normalize(vmin=-0.025, vmax=0.025)
        cm = matplotlib.cm.get_cmap('RdBu_r')
        #plt.scatter(-0.15+0*y, y, c=cm(norm(y)), marker=0, zorder=1)
        #plt.scatter(-0.115+0*y, y, c=cm(norm(-y)), marker=0, zorder=0)
        plt.xlim(xl)
        ax.set_ylabel("")

        legend(frameon=False)
        sns.despine()



        _figure5A(oglssd, oglidx, gs[3, :])
        add_letter(plt.gcf(), gs[3,:], 'B', y=1.4)
        # _figure5C(ssd.test_slope.Averaged, oglssd.test_slope.Pair, gs=gs[3,:])
        savefig(
            "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/3_figure_3.pdf",
            dpi=1200,
            bbox_inches="tight",
        )

        #plt.figure(figsize=(7.5, 1.5))
        #gs = matplotlib.gridspec.GridSpec(1, 3, hspace=0.0)
        #_figure5B(gs=gs[:, :])
        #savefig(
        #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_2_figure_5.pdf",
        #    dpi=1200,
        #    bbox_inches="tight",
        #)
    return ssd, idx, brain


def _figure5A(ssd, idx, gs, integration_slice=def_ig):
    import seaborn as sns
    import matplotlib

    gs = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs, height_ratios=[1, 0.075])
    with mpl.rc_context(rc=rc):
        dv = 10
        ax = plt.subplot(gs[0, 0])        
        m = idx.groupby("cluster").mean()
        palette = {k: dv + vals.SSD for k, vals in m.iterrows()}
        img = _get_img(palette, low=dv + -0.05, high=dv + 0.05, views=["lat", "med"])
        plt.imshow(img, aspect="equal")        
        plt.xticks([])
        plt.yticks([])
        plt.title("Sample\ncontrast decoding", fontsize=7)
        sns.despine(ax=ax, left=True, right=True, bottom=True)

        cax = plt.subplot(gs[1, 0])
        norm = mpl.colors.Normalize(vmin=0,vmax=0.05)
        sm = plt.cm.ScalarMappable(
            cmap=truncate_colormap(plt.get_cmap("RdBu_r"), 0.5, 1),
            norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal',
            ticks=[0, 0.025, 0.05], drawedges=False, shrink=0.6)
        cbar.ax.set_xticklabels(
            ['0', '0.025', '0.05'], fontsize=7)
        ax = plt.subplot(gs[0, 1])
        m = idx.groupby("cluster").mean()
        palette = {k: dv + vals.SSD_acc_contrast for k, vals in m.iterrows()}
        img = _get_img(palette, low=dv + -0.05, high=dv + 0.05, views=["lat", "med"])
        plt.imshow(img, aspect="equal")

        #plt.colorbar(sm)        
        plt.xticks([])
        plt.yticks([])
        plt.title("Accumulated\ncontrast decoding", fontsize=7)
        sns.despine(ax=ax, left=True, right=True, bottom=True)

        cax = plt.subplot(gs[1, 1])
        norm = mpl.colors.Normalize(vmin=0,vmax=0.05)
        sm = plt.cm.ScalarMappable(
            cmap=truncate_colormap(plt.get_cmap("RdBu_r"), 0.5, 1),
            norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal',
            ticks=[0, 0.025, 0.05], drawedges=False, shrink=0.8)
        cbar.ax.set_xticklabels(
            ['0', '0.025', '0.05'], fontsize=7)
        ax = plt.subplot(gs[0, 2])

        m = idx.groupby("cluster").mean()
        palette = {k: dv + vals.SSDvsACC for k, vals in m.iterrows()}
        img = _get_img(palette, low=dv + -0.025, high=dv + 0.025, views=["lat", "med"])
        plt.imshow(img, aspect="equal")
        norm = mpl.colors.Normalize(vmin=-0.025,vmax=0.025)
        sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
        sm.set_array([])
        #plt.colorbar(sm, orientation='horizontal')
        plt.xticks([])
        plt.yticks([])
        plt.title("Sample - Accumulated\ncontrast decoding", fontsize=7)
        sns.despine(ax=ax, left=True, right=True, bottom=True)
        cax = plt.subplot(gs[1, 2])
        norm = mpl.colors.Normalize(vmin=-0.025,vmax=0.025)
        sm = plt.cm.ScalarMappable(
            cmap="RdBu_r",
            norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal',
            ticks=[-0.025, 0, 0.025], drawedges=False, shrink=0.8)
        cbar.ax.set_xticklabels(
            ['-0.025\naccumulated\ncontrast enc.', '0', '0.025\nsample\ncontrast enc.'], fontsize=7)
    return ssd, idx


def figureS4(xscores=None, gs=None):
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
        figure(figsize=(7.5, 1.5))
        gs = matplotlib.gridspec.GridSpec(1, 5, width_ratios=[1, 0.4, 1, 1, 1])
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=gs, width_ratios=[1, 0.35, 1, 1, 1]
        )
    area_names = ["aIPS", "IPS/PostCeS", "M1-hand"]
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
            label="Weighted samples",
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
        draw_sig(ax, np.arctanh(yc - y1smp), p=0.05, fdr=False, lw=2, color=sns.xkcd_rgb[colors[0]])
        #draw_sig(ax, yc, fdr=False, lw=2, color=sns.xkcd_rgb[colors[0]])
        draw_sig(ax, np.arctanh(yint - y1smp), p=0.05, fdr=False, lw=2, y=0.0015, color=sns.xkcd_rgb[colors[2]])
       # draw_sig(ax, yint, fdr=False, lw=2, y=0.0015, color=sns.xkcd_rgb[colors[2]])
        
        title(area_names[i], fontsize=7, color=area_colors[signal])
        if i == 0:
            ax.set_xlabel("Time after sample onset")
            ax.set_ylabel("Correlation")
            sns.despine(ax=ax)
        elif i == 2:            
            ax.legend(frameon=False, loc='center', bbox_to_anchor=(1.7, 0.75))
            ax.set_yticks([])
            sns.despine(ax=ax, left=True)
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
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/S4_figure_S4.pdf",
        dpi=1200,
        bbox_inches="tight",
    )


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
    "DCD": '#db4514', #'#01a56c', #figure_6colors[1],
    "AF-CIFcorr": figure_6colors[2],
    "lAFcorr": figure_6colors[0],
    "lAF-CIFcorr": figure_6colors[2],
    "AccDecode":'#6c01a5'
}


def figure6():
    plt.figure(figsize=(8, 4.6))
    gs = matplotlib.gridspec.GridSpec(2, 3, height_ratios=[1, 0.8], hspace=0.25)
    with mpl.rc_context(rc=rc):
        subplot(gs[0,:], zorder=10)
        _figure6_comics(N=10000000)
        add_letter(gcf(), gs[0,:], 'A', y=0.95)
        _figure6A(gs=gs[1, :])
        #_figure6B(gs=gs[2, :])        
        savefig(
            "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/4_figure_4.pdf",
            dpi=1200,
            bbox_inches="tight",
        )


def _figure6_comics(N=1000):        
    with mpl.rc_context(rc=rc):
        ax = cp_sim(N=N)
        #text(
        #    9.5,
        #    0.77,
        #    "No sensory adaptation",
        #    horizontalalignment="center",
        #    verticalalignment="center",
        #)
        for i in list(range(0, 10)) + list(range(10, 20)) + list(range(20, 30)):
            plot([i, i], [0.5, 0.725], color='gray', alpha=0.5, lw=0.5, zorder=-100)
        text(4.5, 0.75, "(i) Constant sensory gain,\nperfect accumulation", horizontalalignment="center")
        text(
            14.5, 0.75, "(ii) Constant sensory gain,\nbounded accumulation", horizontalalignment="center"
        )
        #plot([4.5, 4.5], [0.75, 0.7725], 'k', lw=1)
        #plot([4.5, 6], [0.7725, 0.7725], 'k', lw=1)
        #plot([13, 14.5], [0.7725, 0.7725], 'k', lw=1)
        #plot([14.5, 14.5], [0.75, 0.7725], 'k', lw=1)
        
        #text(
        #    24.5,
        #    0.77,
        #    "With sensory adaptation",
        #    horizontalalignment="center",
        #    verticalalignment="center",
        #)
        text(24.5, 0.75, "(iii) Sensory adaptation,\nperfect accumulation", horizontalalignment="center")
        #text(
        #    34.5, 0.71, "Bounded accumulation", horizontalalignment="center"
        #)   
        shift = 20
        #plot([4.5+shift, 4.5+shift], [0.75, 0.7725], 'k', lw=1)
        #plot([4.5+shift, 5.7+shift], [0.7725, 0.7725], 'k', lw=1)
        #plot([13.3+shift, 14.5+shift], [0.7725, 0.7725], 'k', lw=1)
        #plot([14.5+shift, 14.5+shift], [0.75, 0.7725], 'k', lw=1)
        
        ax.legend(frameon=False, bbox_to_anchor=(1.07, -.15), loc=0)     
        ax.set_xlim([0, 30])

@memory.cache()
def _get_response(
    fluct, bounded=False, early_noise=0.4, late_noise=0, weights=np.array([1] * 10)
):
    N = fluct.shape[0]
    from conf_analysis.behavior import kernels
    def bounded_acc(fluct, bound=0.95):
        responses = fluct[:, 0] * np.nan
        cumsums = fluct.cumsum(1)
        for col in np.arange(fluct.shape[1]):
            id_pos = (cumsums[:, col] > bound) & np.isnan(responses)
            id_neg = (cumsums[:, col] < -bound) & np.isnan(responses)
            responses[id_pos] = 1
            responses[id_neg] = -1
        id_pos = (cumsums[:, col] > 0) & np.isnan(responses)
        id_neg = (cumsums[:, col] < 0) & np.isnan(responses)
        responses[id_pos] = 1
        responses[id_neg] = -1
        return responses

    early_noise = early_noise * (np.random.normal(size=(N, 10)))
    late_noise = early_noise * (np.random.normal(size=(N, 10)))
    correct = fluct.mean(1) > 0

    internal_flucts = (fluct + early_noise) * weights
    if bounded:
        resp = bounded_acc(internal_flucts + late_noise, bound=bounded)
    else:
        resp = ((internal_flucts + late_noise).mean(1) > 0).astype(int)
        resp[resp == 0] = -1
    accuracy = np.mean((resp == 1) == correct)
    behavioral_kernel = kernels.kernel(fluct, resp.astype(int))[0]
    cp = kernels.kernel(internal_flucts, resp.astype(int))[0]
    contrast = weights-weights.mean()
    return accuracy, behavioral_kernel, cp, contrast.ravel()


@memory.cache()
def _get_fluct(t, N):
    return 0.1 * (np.random.normal(size=(N, 10))) + t

def cp_sim(t=0.1, noise_mag=0.43, N=15000):
    fluct = _get_fluct(t, N)
    yl = [0.499, 0.8]
    
    const = np.linspace(1, 0, 10)[np.newaxis, :] * 0 + 1
    weights = np.linspace(1, 0.1, 10)[np.newaxis, :]
    ax = gca()
    for i, (bound, weights, enfudge) in enumerate(
        zip([False, True, False,],#True], 
            [const, const, weights,],# weights], 
            [0.43, 0.385, 0.286,])# 0.295])
    ):
        x = np.arange(10)+i*10
        
        accuracies = []
        kernels = []
        contrasts = []
        if bound:
            for b in [0.95-0.5, 0.95, 0.95+0.5]:
                accuracy, behavioral_kernel, cp, contrast = _get_response(
                    fluct, bounded=b, weights=weights, early_noise=enfudge                
                )
                accuracies.append(accuracy)
                kernels.append(behavioral_kernel)
                contrasts.append(contrast)
        elif weights is const:

            accuracy, behavioral_kernel, cp, contrast = _get_response(
                fluct, bounded=bound, weights=weights, early_noise=enfudge
            )
            accuracies.append(accuracy)
            kernels.append(behavioral_kernel)
            contrasts.append(contrast)
        else:
            for w in [0.75, 1, 1.25]:
                ws = np.linspace(w, 0.1, 10)[np.newaxis, :]
                accuracy, behavioral_kernel, cp, contrast = _get_response(
                    fluct, bounded=bound, weights=ws, early_noise=enfudge
                )
                accuracies.append(accuracy)
                kernels.append(behavioral_kernel)
                contrasts.append(contrast)
        
        for j,(behavioral_kernel, accuracy, contrast) in enumerate(zip(kernels, accuracies, contrasts)):
            lw = 1
            if (len(kernels) == 3) and not (j==1):
                lw = 0.5
            if i==0:
                plot(x, behavioral_kernel+0.05, lw=lw, color="k", label="Behavioral kernel")
                #plot(x, cp, color=figure_6colors["AFcorr"], label="V1 (gamma band) kernel")
                if contrast[-1] < 0:
                    plot(x, 0.1*contrast+0.6, lw=lw, color=figure_6colors["DCD"], label="V1 contrast")
                else:
                    plot(x, contrast+0.6, lw=lw, color=figure_6colors["DCD"], label="V1 contrast")
            else:
                plot(x, behavioral_kernel+0.05, lw=lw, color="k")
                #plot(x, cp, color=figure_6colors["AFcorr"])
                if contrast[-1] < 0:
                    plot(x, 0.1*contrast+0.6, lw=lw, color=figure_6colors["DCD"])
                else:
                    plot(x, contrast+0.6, lw=lw, color=figure_6colors["DCD"])
        ylim(yl)
        #text(0+i*10+0.5, 0.509, 'Accuracy=%0.3f'%np.around(accuracy, 3))
        text(0+i*10+4.5, 0.46, 'Time (s)', horizontalalignment='center')
        yticks([])
    #xticks([0, 9, 10, 19, 20, 29, 30, 39], [0, 1, 0, 1, 0, 1, 0, 1])
    xticks([0, 9, 10, 19, 20, 29,], [0, 1, 0, 1, 0, 1,])
    sns.despine(ax=ax, left=True, bottom=True)
    ylabel('AUC /      \nDecoding precision (a.u)')
    plot([0, 0], [0.5, 0.725], 'k')
    plot([0, 9], [0.5, 0.5], 'k')
    plot([10, 19], [0.5, 0.5], 'k')
    plot([20, 29], [0.5, 0.5], 'k')
    #plot([30, 39], [0.5, 0.5], 'k')
    return ax


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
            1, 3, width_ratios=[1, 1, 0.1], wspace=0.35
        )
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs, width_ratios=[1, 1, 0.1], wspace=0.35
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
    draw_sig(ax, K, y=-0.001, fdr=True, color="k")
    sns.despine(ax=ax, bottom=False, right=True)
    plt.ylabel("AUC-0.5")
    plt.xticks(np.arange(10), ["0"] + [""] * 8 + ["1"])
    plt.xlabel("Time (s)")
    plt.yticks([0, 0.04, 0.08, 0.12])
    plt.ylim([-0.003, 0.13])
    for i in range(10):
        plt.axvline(i, color='gray', zorder=-100000, alpha=0.5)
    ax = plt.subplot(gs[0, 1])
    plotwerr(v1decoding.T, label="V1 contrast", color=colors["DCD"], lw=1)
    draw_sig(ax, K, y=-0.001, fdr=True, color=colors["DCD"])
    plt.yticks([0, 0.04, 0.08, 0.12])
    plt.xticks(np.arange(10), ["0"] + [""] * 8 + ["1"])
    plt.xlabel("Time (s)")
    plt.ylabel("Decoding precision (a.u.)")
    plt.ylim([-0.003, 0.13])
    # plt.axhline(0, color='k', lw=1)
    #plt.legend(
    #    frameon=False, ncol=1, loc="center left", bbox_to_anchor=[0.2, 1], fontsize=8
    #)
    for i in range(10):
        plt.axvline(i, color='gray', zorder=-100000, alpha=0.5)
    sns.despine(ax=ax, bottom=False, right=True)
    add_letter(plt.gcf(), gs[:,:2], 'B')

    ax = plt.subplot(gs[0, 2])
    nine = np.array([np.corrcoef(x[:9], y[:9])[0,1] for x,y in zip(K.values, clpeaks['vfcPrimary'].values.T)])
    plot(nine*0 + np.random.randn(15)/5, nine, 'ok', alpha=0.5)
    plot([-0.5, 0.5], [np.mean(nine), np.mean(nine)], 'k', lw=2)
    t,p = ttest_1samp(np.arctanh(nine), 0)
    print('Correlation kernel, decoding (m, t, p):', np.mean(nine), t,p)
    ylim([-1, 1])
    xlim([-0.75, 0.75])
    xticks([])
    yticks([-1, -0.5, 0, 0.5, 1])
    ylabel('Correlation behavioral kernel\nand V1 Contrast kernel')
    sns.despine(ax=ax, bottom=True, right=True)
    add_letter(plt.gcf(), gs[:,2], 'C', x=-2.2)

    """
    ax = plt.subplot(gs[0, 2], zorder=1)
    plotwerr(alls, label="V1 kernel (gamma band)", color=colors["AFcorr"])
    #plotwerr(rems, label="Contrast fluctu-\nations removed", color=colors["AF-CIFcorr"])    
    draw_sig(ax, alls, fdr=False, y=-0.0025, color=colors["AFcorr"])
    #draw_sig(ax, rems, y=-0.01125, color=colors["AF-CIFcorr"])
    #draw_sig(ax, alls - rems, y=-0.0125, color="k")

    # plt.xticks(np.arange(10), np.arange(10)+1)
    plt.xticks(np.arange(10), ["0"] + [""] * 8 + ["1"])
    plt.xlabel("Time")
    plt.ylabel("AUC")
    plt.yticks([0, 0.01, 0.02, 0.03])
    plt.ylim(-0.003, 0.03)
    #yl = plt.ylim()
    # plt.axhline(0, color='k', lw=1)
    #plt.legend(
    #    frameon=False, ncol=1, loc="center left", bbox_to_anchor=[0.15, 0.8], fontsize=8
    #)
    sns.despine(ax=ax, bottom=False, right=True)
    #plt.title("Gamma", fontsize=8)
    """


def figure7():
    plt.figure(figsize=(8, 4.5))
    gs = matplotlib.gridspec.GridSpec(2, 3, height_ratios=[1, 0.8], hspace=0.4)
    with mpl.rc_context(rc=rc):                
        _figure7A(gs=gs[0, :])
        _figure7B(gs=gs[1, :])
        savefig(
            "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/5_figure_5.pdf",
            dpi=1200,
            bbox_inches="tight",
        )


def _figure7B(cluster="vfcPrimary", gs=None):
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

        remsslopes = rems.apply(
            lambda x: linregress(np.arange(10), x)[0], axis=1
        ).to_frame()
        # alls.mean(1).to_frame()
        remsslopes.loc[:, "comparison"] = "rems_slope"
        remsslopes.loc[:, "freq"] = freq
        remsslopes.columns = ["sum", "comparison", "freq"]
        res.append(remsslopes)

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
    xlabel("Frequency (Hz)", fontsize=7)
    ylabel("Kernel sum", fontsize=7)
    yticks([-0.005, 0, 0.005, 0.01])
    #for i in range(10):
    #    ax.axvline(i, color='gray', alpha=0.5, lw=0.5)
    ax.axhline(0, color='gray', zorder=-10)
    
    sns.despine(ax=ax)

    add_letter(plt.gcf(), gs[0,0], 'C', x=-0.35)

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

    #--> REM SLOPES
    diffr = pd.pivot_table(
        data=res.query('comparison=="rems_slope"'),
        index="subject",
        columns="freq",
        values="sum",
    )

    plotwerr(diffr, color=figure_6colors["AF-CIFcorr"])
    print('--->',ttest_1samp(diffr.values, 0)[1])
    # print(diff.values.mean(0), p)
    #print("Slope sig. frequencies:", diff.columns.values[p < 0.05])
    draw_sig(ax, diffr, fdr=True, color='r', y=-0.0002)
    #draw_sig(ax, diff-diffr, fdr=True, color='k', y=0.0002, zorder=-10)

    #<-- REM SLOPES
    # draw_sig(ax,diff, fdr=True, color='g')
    xlabel("Frequency (Hz)", fontsize=7)
    ylabel("Slope of\nV1 kernel", fontsize=7)
    yticks([-0.002, 0, 0.002])
    sns.despine(ax=ax)
    add_letter(plt.gcf(), gs[0,2], 'D', x=-0.4)
    #for i in range(10):
    #    ax.axvline(i, color='gray', alpha=0.5, lw=0.5)
    ax.axhline(0, color='gray', zorder=-10)

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
    print("P-values Feedback M1->V1:", ttest_1samp(np.arctanh(alpha_M1), 0))
    draw_sig(ax, np.arctanh(alpha_M1), color=figure_6colors["AFcorr"])
    ylabel("Correlation\nV1 Alpha /  M1 decoding", fontsize=7)
    yticks([0, 0.15, 0.3], fontsize=7)
    xlabel("Lag (number of\nstimulus samples)", fontsize=7)
    xticks([-2, 0, 2], ["-2\nV1 leads", 0, "2\nM1 leads"], fontsize=7)
    axvline(0, color='gray', zorder=-10)
    sns.despine(ax=ax)
    add_letter(plt.gcf(), gs[0,-1], 'E', x=-0.8)
    return res, rescorr


def get_kernels_from_file(fname, cluster='vfcPrimary'):
    import pickle
    a = pickle.load(open(fname, "rb"))
    ccs, K, kernels, peaks = a["ccs"], a["K"], a["kernels"], a["v1decoding"]
    kernels.set_index(["cluster", "rmcif", "subject"], inplace=True)
    KK = np.stack(kernels.kernel)
    kernels = pd.DataFrame(KK, index=kernels.index).query('cluster=="%s"' % cluster)
    rems = pd.pivot_table(data=kernels.query("rmcif==True"), index="subject")
    alls = pd.pivot_table(data=kernels.query("rmcif==False"), index="subject")
    return rems, alls


def _figure7A(cluster="vfcPrimary", freq="gamma", lowfreq=10, gs=None, label=True):
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
    print('Lowfreq filename:', fname)
    b = pickle.load(open(fname, "rb"))
    low_kernels = b["kernels"]

    colors = figure_6colors

    
    if gs is None:
        figure(figsize=(10.5, 3))
        gs = matplotlib.gridspec.GridSpec(
            1, 8, width_ratios=[1, 0.5, 0.25, 1, 1, 0.25, 0.5, 0.5]
        )
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 8, subplot_spec=gs, width_ratios=[1, 0.75, 0.25, 1, 1, 0.55, 0.35, 0.35]
        )
    

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

    _, lowvar_gamma_alls = get_kernels_from_file("/Users/nwilming/u/conf_analysis/results/cort_kernel.results_varlow_f[45-65].pickle")
    _, medvar_gamma_alls = get_kernels_from_file("/Users/nwilming/u/conf_analysis/results/cort_kernel.results_varmed_f[45-65].pickle")
    _, highvar_gamma_alls = get_kernels_from_file("/Users/nwilming/u/conf_analysis/results/cort_kernel.results_varhigh_f[45-65].pickle")

    _, lowvar_alpha_alls = get_kernels_from_file("/Users/nwilming/u/conf_analysis/results/cort_kernel.results_varlow_f[0-10].pickle")
    _, medvar_alpha_alls = get_kernels_from_file("/Users/nwilming/u/conf_analysis/results/cort_kernel.results_varmed_f[0-10].pickle")
    _, highvar_alpha_alls = get_kernels_from_file("/Users/nwilming/u/conf_analysis/results/cort_kernel.results_varhigh_f[0-10].pickle")

    ax = plt.subplot(gs[0, :3], zorder=2)
    
    #plotwerr(lowvar_gamma_alls, label="V1 kernel (low)", color="r")
    #plotwerr(medvar_gamma_alls, label="V1 kernel (medium)", color='g')
    #plotwerr(highvar_gamma_alls, label="V1 kernel (high)", color='b')
    plotwerr(alls, label="Overall kernel", color=colors["AFcorr"])
    plotwerr(rems, label="Residual kernel (contrast\nfluctuations removed)", color=colors["AF-CIFcorr"])#, fontsize=7)

    gamma_k_slopes = alls.T.apply(lambda x: linregress(np.arange(0, 1, 0.1), x)[0])
    print('Mean Gamma kernel slope:', gamma_k_slopes.mean(), ttest_1samp(gamma_k_slopes, 0))
    draw_sig(ax, alls, y=-0.01, color=colors["AFcorr"])
    draw_sig(ax, rems, y=-0.01125, color=colors["AF-CIFcorr"])
    draw_sig(ax, alls - rems, y=-0.0125, color="k")
    # plt.xticks(np.arange(10), np.arange(10)+1)
    plt.xticks(np.arange(10), ["1"] + [""] * 8 + ["10"], fontsize=7)
    plt.xlabel("Contrast sample number", fontsize=7)
    plt.ylabel("AUC-0.5", fontsize=7)
    plt.yticks([0, 0.02, 0.05], fontsize=7)
    yl = plt.ylim()
    # plt.axhline(0, color='k', lw=1)
    plt.legend(
        frameon=False, ncol=1, loc="center left", bbox_to_anchor=[0.15, 0.8], fontsize=7
    )
    for i in range(10):
        ax.axvline(i, color='gray', alpha=0.5, lw=0.5, zorder=-100)
    ax.axhline(0, color='gray', zorder=-100)
    sns.despine(ax=ax, bottom=False, right=True)
    plt.title("Gamma (45-60Hz)", fontsize=7)

    add_letter(plt.gcf(), gs[0,:3], 'A', x=-0.3)

    ax = plt.subplot(gs[0, 3:5], zorder=1)

    plotwerr(low_alls, ls=":", color=colors["AFcorr"])  # label='V1 kernel',
    plotwerr(low_rems, ls=":", color=colors["AF-CIFcorr"])
    
    #plotwerr(lowvar_alpha_alls, label="V1 kernel (low)", color="r")
    #plotwerr(medvar_alpha_alls, label="V1 kernel (medium)", color='g')
    #plotwerr(highvar_alpha_alls, label="V1 kernel (high)", color='b')

    draw_sig(ax, low_alls, y=-0.01, color=colors["AFcorr"])
    draw_sig(ax, low_rems, y=-0.01125, color=colors["AF-CIFcorr"])
    draw_sig(ax, low_alls - low_rems, y=-0.0125, color="k")

    ## Do RM anova for alpha kernels
    print('T-test, high vs. low var:', ttest_rel(lowvar_alpha_alls.loc[:, 7], highvar_alpha_alls.loc[:, 7]))
    import pingouin as pg
    lowvar_alpha_alls.columns.name = 'sample'
    lowvar_alpha_alls = lowvar_alpha_alls.stack().to_frame()
    lowvar_alpha_alls.loc[:, 'variance'] = 0.05
    lowvar_alpha_alls.set_index(['variance'], append=True, inplace=True)
    lowvar_alpha_alls.columns = ['auc']
    
    medvar_alpha_alls.columns.name = 'sample'
    medvar_alpha_alls = medvar_alpha_alls.stack().to_frame()
    medvar_alpha_alls.loc[:, 'variance'] = 0.1
    medvar_alpha_alls.set_index(['variance'], append=True, inplace=True)
    medvar_alpha_alls.columns = ['auc']

    highvar_alpha_alls.columns.name = 'sample'
    highvar_alpha_alls = highvar_alpha_alls.stack().to_frame()
    highvar_alpha_alls.loc[:, 'variance'] = 0.15
    highvar_alpha_alls.set_index(['variance'], append=True, inplace=True)
    highvar_alpha_alls.columns = ['auc']


    alpha_by_var = pd.concat([lowvar_alpha_alls, medvar_alpha_alls, highvar_alpha_alls]).reset_index()
    
    aov = pg.rm_anova(dv='auc', within=['variance', 'sample'],
                   subject='subject', data=alpha_by_var.reset_index(), detailed=True)
    print('Within subject RM-Anova for kernel variance:')
    print(aov.round(2))
    plt.legend(
        frameon=False, ncol=1, fontsize=7
    )

    for i in range(10):
        ax.axvline(i, color='gray', alpha=0.5, lw=0.5, zorder=-100)
    ax.axhline(0, color='gray', zorder=10)
    plt.xticks(np.arange(10), ["1"] + [""] * 8 + ["10"], fontsize=7)
    plt.xlabel("Contrast sample number", fontsize=7)
    plt.ylim(yl)
    plt.ylabel(None)
    plt.yticks([])
    plt.title("Alpha (10Hz)", fontsize=7)
    # plt.axhline(0, color='k', lw=1)

    sns.despine(ax=ax, bottom=False, right=True, left=True)

    rescorr = []
    for i in range(15):
        rescorr.append(
            {
                "AFcorr": np.corrcoef(alls.loc[i + 1], K.loc[i + 1, :])[0, 1],
                "AFcorr_nine_samples": np.corrcoef(alls.loc[i + 1, :8], K.loc[i + 1, :8])[0, 1],
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
        #import pdb; pdb.set_trace()
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
        order=["AFcorr"],# "AF-CIFcorr"],
        palette='gray',
    )

    plt.ylabel("Correlation with\nbehavioral kernel", fontsize=7)
    plt.plot(
        [-0.2, +0.2],
        [rescorr.loc[:, "AFcorr"].mean(), rescorr.loc[:, "AFcorr"].mean()],
        "k", lw=2, 
        zorder=100,
    )
    p = ttest_1samp(np.arctanh(rescorr.loc[:, "AFcorr"]), 0)
    print(
        "M/T/P corr gamma kernel w/ behavior kernel:",
        np.around(np.mean(rescorr.loc[:, "AFcorr"]), 2),
        p,
    )
    p9 = ttest_1samp(np.arctanh(rescorr.loc[:, "AFcorr_nine_samples"]), 0)
    print(
        "M/T/P corr gamma kernel w/ behavior kernel, first nine samples only:",
        np.around(np.mean(rescorr.loc[:, "AFcorr_nine_samples"]), 2),
        p9,
    )
    if p[1] < 0.001:
        plt.text(0, 0.9, '*', fontsize=12, verticalalignment='center', horizontalalignment='center')


        #plt.plot(
        #    [-0.125, +0.125], [-0.95, -0.95], color=colors["AFcorr"], zorder=100, lw=2
        #)

    #plt.plot(
    #    [1 - 0.2],# 1 + 0.2],
    #    [rescorr.loc[:, "AF-CIFcorr"].mean()],# rescorr.loc[:, "AF-CIFcorr"].mean()],
    #    "k", lw=2,
    #    zorder=100,
    #)
    p = ttest_1samp(np.arctanh(rescorr.loc[:, "AF-CIFcorr"]), 0)
    print(
        "M/T/P corr gamma kernel -CIF w/ behavior kernel:",
        np.around(np.mean(rescorr.loc[:, "AF-CIFcorr"]), 2),
        p,
    )
    if p[1] < 0.05:
        plt.text(1, 0.95, '*', fontsize=10, verticalalignment='center')
        #plt.plot(
        #    [1 - 0.125, 1 + 0.125],
        #    [-0.95, -0.95],
        #    color=colors["AF-CIFcorr"],
        #    zorder=100,
        #    lw=2,
        #)
    p = ttest_rel(
        np.arctanh(rescorr.loc[:, "AF-CIFcorr"]), np.arctanh(rescorr.loc[:, "AFcorr"])
    )
    print(
        "M/T/P corr gamma kernel -CIF w/bK vs corr gamma kernel w/bK:",
        np.around(np.mean(rescorr.loc[:, "AF-CIFcorr"] - rescorr.loc[:, "AFcorr"]), 2),
        p,
    )
    plt.title("Gamma", fontsize=7)
    plt.xlabel("")
    plt.xticks([])
    plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize=7)
    plt.ylim([-1, 1])
    plt.xlim(-0.3, 0.3)
    yl = plt.ylim()
    xl = plt.xlim()
    #plt.xlim([xl[0]-0.2, xl[1]])
    sns.despine(ax=ax, bottom=True)

    add_letter(plt.gcf(), gs[0,-2], 'B', x=-1.2)

    ax = plt.subplot(gs[0, -1])
    sns.stripplot(
        data=rs,
        x="comparison",
        y="correlation",
        order=["lAFcorr"],# "lAF-CIFcorr"],
        #palette=colors,
        palette='gray'
    )
    plt.ylim(yl)
    plt.plot(
        [-0.2, 0 + 0.2],
        [rescorr.loc[:, "lAFcorr"].mean(), rescorr.loc[:, "lAFcorr"].mean()],
        "k", lw=2,
        zorder=100,
    )
    p = ttest_1samp(np.arctanh(rescorr.loc[:, "lAFcorr"]), 0)
    print(
        "M/T/P corr alpha kernel -CIF w/ behavior kernel:",
        np.around(np.mean(rescorr.loc[:, "lAFcorr"]), 2),
        p,
    )
    if p[1] < 0.05:
        plt.text(0, 0.9, '*', fontsize=12, verticalalignment='center')
        #plt.plot(
        #    [-0.125, +0.125], [-0.95, -0.95], color=colors["AFcorr"], zorder=100, lw=2
        #)
    #plt.plot(
    #    [1 - 0.2],# 1 + 0.2],
    #    [rescorr.loc[:, "lAF-CIFcorr"].mean()],# rescorr.loc[:, "lAF-CIFcorr"].mean()],
    #    "k", lw=2,
    #    zorder=100,
    #)
    p = ttest_1samp(np.arctanh(rescorr.loc[:, "lAF-CIFcorr"]), 0)
    print(
        "M/T/P corr alpha kernel -CIF w/ behavior kernel:",
        np.around(np.mean(rescorr.loc[:, "AF-CIFcorr"]), 2),
        p,
    )
    if p[1] < 0.05:
        plt.text(1, 0.9, '*', fontsize=12, verticalalignment='center')
        #plt.plot(
        #    [1 - 0.125, 1 + 0.125],
        #    [-0.95, -0.95],
        #    color=colors["AF-CIFcorr"],
        #    zorder=100,
        #    lw=2,
        #)
    p = ttest_rel(
        np.arctanh(rescorr.loc[:, "lAF-CIFcorr"]), np.arctanh(rescorr.loc[:, "lAFcorr"])
    )
    print(
        "M/T/P corr alpha kernel -CIF w/bK vs corr alpha kernel w/bK:",
        np.around(np.mean(rescorr.loc[:, "AF-CIFcorr"] - rescorr.loc[:, "lAFcorr"]), 2),
        p,
    )
    plt.title("Alpha", fontsize=7)
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
    plt.xlim(-0.3, 0.3)
    sns.despine(ax=ax, bottom=True, left=True)

    #ax = plt.subplot(gs[0, 1], zorder=-1)
    #pal = {}
    #for cluster in peaks.columns.get_level_values("cluster").unique():
    #    r = []
    #    for i in range(15):
    #       r.append(np.corrcoef(peaks[cluster].T.loc[i + 1], K.loc[i + 1, :])[0, 1])
    #    pal[cluster] = np.mean(r)
    #
    # pal = dict(ccs.groupby('cluster').mean().loc[:, 'AFcorr'])
    #img = _get_lbl_annot_img(
    #    {k: v + 1 for k, v in pal.items()}, low=1 + -1, high=1 + 1, thresh=-1
    #)
    #ax.imshow(img)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #sns.despine(ax=ax, left=True, bottom=True)
    return rescorr


def plot_per_sample_resp(ax, ssd, area, area_label, integration_slice, ylim=None, 
    hull=False, acc=True, sig=False):
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

    if sig:
        for sample in range(1, 10):
            # print(area, (samples_k[sample]-samples_ka[sample]).loc[:, 0.1*sample:].head())
            draw_sig(
                ax,
                (samples_k[sample] - samples_ka[sample]).loc[:, 0.1 * sample :],
                fdr=True,
                color="k",
            )

    if hull:
        plt.plot(ka.index, ka.max(1), color=figure_6colors['AccDecode'], label="Accumulated\ncontrast")
        plt.plot(k.index, k.max(1), color=figure_6colors["DCD"], label="Sample\nContrast")
    else:
        #plt.plot(ka.index, ka, color=figure_6colors['AccDecode'], lw=0.1)
        cvals = array([0.71, 0.33, 0.53, 0.75, 0.59, 0.57, 0.55, 0.61, 0.45, 0.58])        
        
        norm=matplotlib.colors.Normalize(-5, 10)
        cm = matplotlib.cm.get_cmap('BuPu')
        colors = [cm(norm(10-i)) for i, c in enumerate(cvals)]
        colors = sns.color_palette('gray', n_colors=12)        
        for i in range(k.shape[1]):            
            plt.plot(k.index, k.loc[:, i], color=colors[i], lw=0.5)
    

    plt.ylim(ylim)
    plt.legend(fontsize=7)
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
    plt.title(area_label, fontsize=7)
    plt.xlabel("Time", fontsize=7)
    plt.ylabel("Decoding precision (a.u.)", fontsize=7)


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


def _supp_leakage():#pairs):
    fig=plt.figure(figsize=(7.5, 7.5))
    import pickle

    area_names = {
        #"vfcPrimary":'V1',
        "vfcEarly":'V2-V4',
        "vfcV3ab":'V3A/B',
        "vfcIPS01":'IPS0/1',
        "vfcIPS23":'IPS2/3',        
        'vfcLO': "LO1/2",
        'vfcTO': "MT/MST",
        'vfcPHC': "PHC",         
        'vfcVO': "VO1/2",
        #"JWG_aIPS",
        #"JWG_IPS_PCeS",
        #"JWG_M1",
    }
    areas = list(area_names.keys())
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
    colors = sns.color_palette('viridis', n_colors=2)
    
    
    
    with mpl.rc_context(rc=rc):
        gs = matplotlib.gridspec.GridSpec(1, 1,) #height_ratios=[1.2, 1])

        gs_a = matplotlib.gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs[0, 0])
        for i, cluster in enumerate(areas):
            #if ('M1' in cluster) or ('aIPS' in cluster) or ('IPS_P' in cluster):
            #    continue
            KK10 = np.stack(kernels10.kernel)
            kk10 = pd.DataFrame(KK10, index=kernels10.index).query(
                'cluster=="%s"' % cluster
            )
            alls10 = pd.pivot_table(data=kk10.query("rmcif==True"), index="subject")
            KK55 = np.stack(kernels55.kernel)
            kk55 = pd.DataFrame(KK55, index=kernels55.index).query(
                'cluster=="%s"' % cluster
            )
            alls55 = pd.pivot_table(data=kk55.query("rmcif==True"), index="subject")
            ax= plt.subplot(gs_a[i // 2, np.mod(i, 2)])
            

            plotwerr(alls10, label="Residual kernel, 10Hz", color=colors[1], linestyle=':')
            draw_sig(ax, alls10,y=-0.008, fdr=False, color=colors[1], linestyle=':')
            plotwerr(alls55, label="Residual kernel, 55Hz", color=colors[1])
            draw_sig(ax, alls55, y=-0.01, fdr=False, color=colors[1])

            plt.title(area_names[cluster], fontsize=7)
            plt.ylim([-0.015, 0.035])
            plt.axhline(0, color='k', zorder=-1, lw=1)
            if np.mod(i, 2) > 0:
                plt.yticks([-0.01, 0, 0.01, 0.02, 0.03], [], fontsize=7)
            else:
                plt.yticks([-0.01, 0, 0.01, 0.02, 0.03], [-0.01, 0, 0.01, 0.02, 0.03], fontsize=7)
                plt.ylabel('AUC-0.5', fontsize=7)
            if (i//2) > 0:
                plt.xticks([0, 4, 9], [1, 5, 10], fontsize=7)
                plt.xlabel('Sample', fontsize=7)
            else:
                plt.xticks([0, 4, 9], [], fontsize=7)
            for x in range(10):
                plt.axvline(x, color='k', lw=1, zorder=-10, alpha=0.5)
            sns.despine(ax=ax)
        plt.legend(fontsize=7, frameon=False)
        #add_letter(fig, gs_a[:,:], 'A', x=-0.071, y=1.1)
        
        """
        # Compute pairwise correlations between kernels
        pairs = []
        for i, c1 in enumerate(areas):        
            for j, c2 in enumerate(areas):            
                for F, d in zip([10, 55], [kernels10, kernels55]):
                    da = np.stack(d.kernel)
                    c1_k = pd.DataFrame(da, index=d.index).query(
                        'cluster=="%s"' % c1
                    )
                    c1_k = pd.pivot_table(data=c1_k.query("rmcif==True"), index="subject")
                    # c1_k Should be num_sub x num_samples

                    db = np.stack(d.kernel)
                    c2_k = pd.DataFrame(db, index=d.index).query(
                        'cluster=="%s"' % c2
                    )
                    c2_k = pd.pivot_table(data=c2_k.query("rmcif==True"), index="subject")                
                    for subject in c1_k.index:
                        pairs.append(
                            {
                                'freq':F, 
                                'c1':c1, 
                                'c2':c2, 
                                'subject':subject,
                                'corr':np.corrcoef(c1_k.loc[subject], c2_k.loc[subject])[0,1]
                            })
        
        pairs = pd.DataFrame(pairs)
        
        gs_c = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0])
        ax = plt.subplot(gs_c[0, 0])
        X55 = pd.pivot_table(
                data=pairs.query("freq==55"), index="c1", columns="c2", values="corr"
            )
            #+ pd.pivot_table(
            #    data=pairs.query("freq==55"), index="c1", columns="c2", values="corr"
            #)
        #) / 2    
        X55 = X55.loc[areas, areas]    
        plt.imshow(X55, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(np.arange(len(X55)))
        ax.set_xticklabels([area_names[x] for x in X55.columns], rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(X55)))
        #ax.set_yticklabels(X55.index, fontsize=7)
        ax.set_yticklabels([area_names[x] for x in X55.index], fontsize=7)
        plt.title("55HZ", fontsize=7)
        ax = plt.subplot(gs_c[0, 1])
        X55 = pd.pivot_table(
                data=pairs.query("freq==10"), index="c1", columns="c2", values="corr"
            )
        #    + pd.pivot_table(
        #        data=pairs.query("freq==15"), index="c1", columns="c2", values="corr"
        #    )
        #) / 2
        X55 = X55.loc[areas, areas]
        plt.imshow(X55, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        plt.title("10HZ", fontsize=7)
        ax.set_xticks(np.arange(len(X55)))
        ax.set_xticklabels([area_names[x] for x in X55.columns], rotation=90, fontsize=7)
        ax.set_yticklabels([])
        cbar = plt.colorbar()            
        cbar.ax.set_yticks([-1, 0, 1])
        #cbar.ax.set_yticklabels(
        #    ['-1', '0\nCorrelation of con-\ntrast removed kernel', '1'], fontsize=7, rotation=90)
        cbar.ax.set_ylabel('Correlation of\nresidual kernel', rotation=90, fontsize=7)
        #sns.despine(ax=ax)
        add_letter(fig, gs_c[:,:], 'B', x=-0.071, y=1.1)
        plt.tight_layout()
        """
    plt.savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/S4_figure_S4.pdf"
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


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    import matplotlib.colors as mcolors
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_auc_sanity(aucs=None, cluster='JWG_IPS_PCeS'):
    if aucs is None:
        aucs = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/auc_sanity_check.hdf')
    auc = (aucs.query('cluster=="%s"'%cluster)
        .groupby(['confidence', 'response'], sort=False)
        .mean())
    aucv = auc# np.triu(auc)
    print(aucv)
    #aucv[aucv==0] = np.nan
    #print(auc)
    #print(np.tril(auc, -1))
    pcolormesh([0, 0.25, 0.5, 0.75,1], [0, 0.25, 0.5, 0.75,1][::-1], aucv, 
        cmap='RdBu_r', vmin=.35, vmax=0.65) 
        #aspect='equal') #extent=[0,1,0,1]
    plot([0, 0.5], [1, 1], 'b', lw=4, label='Resp: Reference')
    plot([.5, 1], [1, 1], 'g', lw=4, label='Resp: Test stim.')

    dy=0.0225
    plot([0, 0.25], [1.+dy, 1.+dy], 'k', lw=4, label='High confidence')
    plot([0.25, 0.5], [1+dy, 1+dy], color=(0.5, 0.5, 0.5), lw=4, label='Low confidence')
    plot([0.5, 0.75], [1+dy, 1+dy], color=(0.5, 0.5, 0.5), lw=4)
    plot([0.75, 1], [1+dy, 1+dy], 'k', lw=4)
    #plot([0, 0.5], [0, 0], 'r', lw=4)
    #plot([.5, 1], [0, 0], 'g', lw=4)
    plot([1, 1], [0, 0.5],  'g', lw=4)
    plot([1, 1], [.5, 1], 'b', lw=4)

    dy=0.015
    plot([1.+dy, 1.+dy],[0, 0.25], 'k', lw=4)
    plot([1+dy, 1+dy], [0.25, 0.5],  color=(0.5, 0.5, 0.5), lw=4)
    plot([1+dy, 1+dy], [0.5, 0.75], color=(0.5, 0.5, 0.5), lw=4)
    plot([1+dy, 1+dy], [0.75, 1], 'k', lw=4)
    #plot([0, 0], [0, 0.5],  'g', lw=4)
    #plot([0, 0], [.5, 1], 'r', lw=4)
    ylim([-.1,1.1])
    xlim([-.1,1.1])
    xticks([])
    yticks([])
    sns.despine(left=True, bottom=True)


def plot_coupling(subject=None, latency=0.1, motor_lat=1.1):
    df = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/all_couplings_new_par.hdf')
    df = df.set_index(['motor_latency', 'readout_latency', 'subject']).stack().to_frame().reset_index()
    df.columns = ['motor_latency', 'readout_latency', 'subject', 'type', 'correlation']
    df.loc[:, 'readout_latency'] = np.around(df.readout_latency, 4)
    df.loc[:, 'motor_latency'] = np.around(df.motor_latency, 4)
    latencies = df.readout_latency.unique()
    latency = latencies[np.argmin(np.abs(latencies-latency))]
    print('Latency:', latency)

    latencies = df.motor_latency.unique()
    motor_lat = latencies[np.argmin(np.abs(latencies-motor_lat))]
    print('Motor Latency:', motor_lat)    

    if subject is not None:
        df = df.query('subject==%i'%subject)

    with mpl.rc_context(rc=rc):
        plt.figure(figsize=(10, 6))
        gs = matplotlib.gridspec.GridSpec(5, 2, width_ratios=[1, 0.5], 
            height_ratios=[1.5, 0.5, 0.8, 0.8, 0.8], )
        ax = subplot(gs[0,0])
        sns.lineplot(x='motor_latency', y='correlation', hue='type', 
            data=(
                df.query('readout_latency==%f'%latency)
                  .groupby(['motor_latency', 'type'])
                  .mean()
                  .reset_index())
        )
        plt.axvline(motor_lat, color='k', alpha=0.5,ls=':')
        plt.xlim([0,1.2])
        sns.despine()
        legend(loc=1, bbox_to_anchor=(1.6,1), frameon=False)
        for i, tpe in enumerate(['weighted_score', 'integrator_score', 'last_sample']):
            ax = subplot(gs[i+2,0])

            K = pd.pivot_table(df.query('type=="%s"'%tpe), 
                columns='readout_latency', 
                index='motor_latency', 
                values='correlation')
            ml = K.index.values
            rl = K.columns.values
            imshow(np.flipud(K.T), 
                extent=[ml.min(), ml.max(), rl.min(), rl.max()], 
                aspect='auto',
                cmap='RdBu_r',
                vmin=-0.1, 
                vmax=0.1)
            sns.despine(ax=ax, left=True, bottom=True)
            if i==2:
                xticks([0, 1])
                xlabel('Time of\nmotor readout')
            else:
                xticks([])
            yticks([])
            plt.arrow( 0, latency, 0.01, 0, fc="k", ec="k",
                head_width=0.025, head_length=0.025 )
            plt.xlim([0,1.2])
            plt.title(tpe, fontsize=7)
        sns.despine(ax=ax, left=False, bottom=False)
        yticks([0, 0.1, 0.2])
        ylabel('V1 readout')

        ax = subplot(gs[2:,1])
        sns.lineplot(x='readout_latency', y='correlation', hue='type', 
            data=(
                df.query('motor_latency==%f'%motor_lat)
                  .groupby(['readout_latency', 'type'])
                  .mean()
                  .reset_index())
        )
        #plt.xlim([0,1.2])
        plt.yticks([0, 0.05])
        plt.axvline(latency, color='k', alpha=0.5,ls=':')
        sns.despine(ax=ax, left=False, bottom=False)
        legend('', frameon=False)
    #plt.savefig('/Users/nwilming/Desktop/coupling_new.pdf')
    plt.figure(figsize=(10, 3))
    
    for sub, d in df.query('type=="integrator_score"').groupby('subject'):
        ax = subplot(4,4,sub)
        K = pd.pivot_table(d, 
            columns='readout_latency', 
            index='motor_latency', 
            values='correlation')
        ml = K.index.values
        rl = K.columns.values
        imshow(np.flipud(K.T), 
            extent=[ml.min(), ml.max(), rl.min(), rl.max()], 
            aspect='auto',
            cmap='RdBu_r',
            vmin=-0.15, 
            vmax=0.15)
        xticks([])
        yticks([])
        ylabel('S%i'%sub)
        sns.despine(ax=ax, left=True, bottom=True)
        #plt.savefig('/Users/nwilming/Desktop/coupling_new_ind_subs.pdf')
        
            
            
            


