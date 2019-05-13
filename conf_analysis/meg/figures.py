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

memory = Memory(location=metadata.cachedir, verbose=-1)

def_ig = slice(0.4, 1.15)


def plotwerr(pivottable, *args, ls='-', label=None, **kwargs):
    N = pivottable.shape[0]
    x =pivottable.columns.values
    mean = pivottable.mean(0).values
    std = pivottable.std(0).values
    sem = std/(N**.5)
    plot(x, mean, *args, label=label, **kwargs)
    if 'alpha' in kwargs:
        del kwargs['alpha']
    if 'color' in kwargs:
        color = kwargs['color']
        del kwargs['color']
        fill_between(x, mean+sem, mean-sem, facecolor=color, edgecolor='none', alpha=0.5, **kwargs)    
    else:
        fill_between(x, mean+sem, mean-sem, edgecolor='none', alpha=0.5, **kwargs)    
    #for col in pivottable:
    #    sem = pivottable.loc[:, col].std() / pivottable.shape[0] ** 0.5
    #    m = pivottable.loc[:, col].mean()
    #    plot([col, col], [m - sem, m + sem], *args, **kwargs)


def draw_sig(ax, pivottable, color='k', fdr=False, lw=2, **kwargs):
    from scipy.stats import ttest_1samp

    p_sig = ttest_1samp(pivottable, 0)[1]
    if fdr:
        from mne.stats import fdr_correction
        id_sig, _ = fdr_correction(p_sig)
        id_sig = list(id_sig)
    else:
        id_sig = list(p_sig<0.05)
    x = pivottable.columns.values
    d = np.where(np.diff([False] + id_sig + [False]))[0]    
    for low, high in zip(d[0:-1:2], d[1::2]):
        #print(x[low], x[high-1])
        ax.plot([x[low], x[high-1]], [0,0], color=color, lw=lw, **kwargs)    
    return p_sig


def _stream_palette():
    rois = ["vfcPrimary", "vfcEarly", "vfcV3ab",
                "vfcIPS01", "vfcIPS23", "JWG_aIPS", 
                "vfcLO", "vfcTO", "vfcVO", "vfcPHC",         
                "JWG_IPS_PCeS", "JWG_M1"]
    return {roi:color for roi, color in zip(rois, sns.color_palette('viridis', n_colors=len(rois)+1))}



def figure1(data=None):
    from conf_analysis.behavior import empirical
    from conf_analysis import behavior

    color_palette = behavior.parse(behavior.colors)
    if data is None:
        data = empirical.load_data()
    data.loc[:, "choice"] = (data.response + 1) / 2
    data.loc[:, "pconf"] = data.confidence - 1

    figure(figsize=(8, 4))
    gs = matplotlib.gridspec.GridSpec(2, 6)

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
                ((edges[:-1] + np.diff(edges)[0] / 2) - (0.5 + threshold)) / threshold,
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
                ((edges[:-1] + np.diff(edges)[0] / 2) - (0.5 + threshold)) / threshold,
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

    subplot(gs[0, :2])
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
    plot([0, 0], [sem[0] + mean[0], mean[0] - sem[0]], color_palette["Secondary2"][0])
    plot([1, 1], [sem[1] + mean[1], mean[1] - sem[1]], color_palette["Secondary1"][0])
    xticks([0, 1], [r"Low", r"High"])
    ylabel("Accuracy")
    xlabel("Confidence")
    xlim(-0.2, 1.2)
    from scipy.stats import ttest_rel, ttest_1samp

    print('T-Test for accuracy by confidence:', ttest_rel(X.loc[:, 0], X.loc[:, 1]))
    sns.despine(ax=gca())
    subplot(gs[0, 2:4])

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
        scrct.append(linregress(np.arange(5), crct.loc[s,:2.3])[0])
        serr.append(linregress(np.arange(5), err.loc[s,:2.3])[0])
    print('Slopes for confidence vs. evidence (correct: mean, t, p, #>0):', np.mean(scrct), ttest_1samp(scrct, 0), sum(np.array(scrct)>0))
    print('Slopes for confidence vs. evidence (error: mean, t, p, #<0):', np.mean(serr), ttest_1samp(serr, 0), sum(np.array(serr)<0))
    plotwerr(crct, color="g", lw=2, label="Correct")
    plotwerr(err, color="r", lw=2, label="Error")
    legend(frameon=False)

    xticks([1, 2], [r"t", r"2t"])
    xlabel("Evidence discriminability")
    ylabel("Confidence")
    yticks([0.2, 0.3, 0.4, 0.5, 0.6])
    sns.despine(ax=gca())

    subplot(gs[0, 4:])
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
    plotwerr(
        high, color=color_palette["Secondary1"][0], alpha=1, lw=2, label="High confidence"
    )
    low = pd.pivot_table(
        index="snum",
        columns="Evidence discriminability",
        values="accuracy",
        data=dz.query("pconf==0.0"),
    )
    plotwerr(low, color=color_palette["Secondary2"][0], alpha=1, lw=2, label="Low confidence")
    legend(frameon=False)
    xticks([1, 2], [r"t", r"2t"])
    xlabel("Evidence discriminability")
    ylabel("% correct")
    yticks([0.5, 0.75, 1], [50, 75, 100])
    sns.despine(ax=gca())
    tight_layout()

    ax = subplot(gs[1, :3])

    palette = {
        r"$E_{N}^{High}$": color_palette["Secondary1"][0],
        r"$E_{N}^{Low}$": color_palette["Secondary2"][1],
        r"$E_{S}^{Low}$": color_palette["Secondary2"][1],
        r"$E_{S}^{High}$": color_palette["Secondary1"][0],
    }

    k = empirical.get_confidence_kernels(data, contrast_mean=0.5)
    for kernel, kdata in k.groupby('Kernel'):
        kk = pd.pivot_table(index='snum', columns='time', values='contrast', data=kdata)
        plotwerr(kk, color=palette[kernel], lw=2, label="Low confidence")
    #empirical.plot_kernel(k, palette, legend=False)
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
    #legend(frameon=False)    
    xticks(np.arange(10), np.arange(10) + 1)
    ax.axhline(color='k', lw=1)
    ax.set_ylim(-0.04, 0.04)
    sns.despine(ax=ax)

    subplot(gs[1, 3:])
    from conf_analysis.behavior import kernels

    dk = data.groupby(["snum", "side"]).apply(kernels.get_decision_kernel).groupby('snum').mean()
    ck = data.groupby(["snum", "side"]).apply(kernels.get_confidence_kernel).groupby('snum').mean().stack()
    confidence_kernel = ck.groupby(["snum"]).mean()
    confidence_assym = ck.query("response==1.0").droplevel("response") - ck.query(
        "response==-1.0"
    ).droplevel("response")
    scrct, serr = [], []
    for s in range(1, 16):
        scrct.append(linregress(np.arange(10), dk.loc[s,])[0])
        serr.append(linregress(np.arange(10), confidence_kernel.loc[s,])[0])
    print('Slopes choice kernel (correct: mean, t, p, #>0):', np.mean(scrct), ttest_1samp(scrct, 0), sum(np.array(scrct)>0))
    print('Slopes for confidence kernel (error: mean, t, p, #<0):', np.mean(serr), ttest_1samp(serr, 0), sum(np.array(serr)<0))

    plotwerr(dk + 0.5, color="k", label="Decision kernel")
    plotwerr(confidence_kernel + 0.5, color=(0.5, 0.5, 0.5), label="Confidence kernel")
    # plotwerr(confidence_assym+0.5, 'b--', label='Confidence assym.')

    # axhline(0.5, color="k", lw=1)
    ylim([0.49, 0.62])
    ylabel("AUC")
    xlabel("Contrast sample")
    legend(frameon=False)
    yticks([0.5, 0.55, 0.6])
    xticks(np.arange(10), np.arange(10) + 1)
    sns.despine(ax=gca(), bottom=False)
    #savefig(
    #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/1_figure_1.pdf",
    #    bbox_inches="tight",
    #)

    p = data.groupby(['snum']).apply(lambda x: empirical.fit_choice_logistic(x, summary=False))
    p=p.groupby('snum').mean()    
    print('Can predict choice above chance in %i/15 subjects. \nMean choice accuracy is %0.2f'%((sum(p>0.5)), np.around(np.mean(p), 2)))

    p = data.groupby(['snum', 'response']).apply(lambda x: empirical.fit_conf_logistic(x, summary=False))
    p=p.groupby('snum').mean()    
    print('Can predict confidence above chance in %i/15 subjects. \nMean confidence accuracy is %0.2f'%((sum(p>0.5)), np.around(np.mean(p), 2)))
    return ck


def figure2(df=None, stats=False):
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
            title_palette=palette
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
            label="% change",
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


def figure3(df=None, stats=False, dcd=None, aspect=0.01883834992799947):
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
        fig = figure(figsize=(7.5, 7.5))
        from conf_analysis.meg import srtfr

        gs = matplotlib.gridspec.GridSpec(2, 1)

        srtfr.plot_stream_figures(
            df.query('~(hemi=="avg")'),
            contrasts=["choice"],
            flip_cbar=True,
            gs=gs[0, 0],
            stats=stats,
            aspect=aspect,
            title_palette=palette
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
            {"MIDC_split": "Reds", "CONF_unsigned": "Blues", "CONF_signed": "Greens"},
            {
                # "Averaged": df.test_roc_auc.Averaged,
                "Lateralized": dcd.test_roc_auc.Lateralized
            },
            gs=gs[1, 0],
            title_palette=palette,
        )
        plotter.plot(aspect="auto")
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/3_figure_3.pdf"
    )
    return df, stats, dcd


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
    gs = matplotlib.gridspec.GridSpec(1, 1) #,height_ratios=[3 / 5, 2 / 5])
    _figure4A(ogldcd, gs=gs[0, 0])
    tight_layout()
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/4_figure_4.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    plt.figure(figsize=(7.5, 4))
    gs = matplotlib.gridspec.GridSpec(1, 1) #, height_ratios=[3 / 5, 2 / 5])
    _figure4B(pdcd, gs=gs[0, 0])
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_4_figure_4.pdf",
        dpi=1200,
        bbox_inches="tight",
    )
    

    return ogldcd, pdcd


def _figure4A(data=None, t=1.117, gs=None):
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
        "MIDC_split": (0.5, 0.65),
        "CONF_signed": (0.5, 0.6),
        "CONF_unsigned": (0.5, 0.55),
    }
    high = 0.65
    low = 0.5
    if gs is None:
        plt.figure(figsize=(7.5, 5))
        gs = matplotlib.gridspec.GridSpec(3, 3)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            4, 3, subplot_spec=gs, hspace=0., height_ratios=[1, 0.25, 1, 1]
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
            interesting_rois = ['M1', 'V1']
            for j, roi in enumerate(sorting):
                val, color = X.loc[roi, t], cm(norm(X.loc[roi, t]))

                plt.plot(X.columns.values, X.loc[roi, :], color=color)

                if X.loc[roi, 1.417] > 0.525:
                    R = roi.replace("JWG_", "").replace("dlpfc_", "")
                    if any([roi in x for x in interesting_rois]):
                        R = re.sub(r'dlpfc.*', 'DLPFC', R)
                        txts.append(plt.text(1.417, X.loc[roi, 1.417] - 0.005, R))
            y = np.linspace(0.475, 0.7, 200)
            x = y * 0 - 0.225
            plt.scatter(x, y, c=cm(norm(y)), marker=0)
            plt.title(titles[signal])
            plt.xlim([-0.25, 1.4])
            plt.ylim([0.475, 0.7])
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
            # brain = dp.plot_brain_color_annotations(palette, low=0.3, high=0.7)
            # 'lateral', 'medial', 'rostral',
            # 'caudal', 'dorsal', 'ventral', 'frontal', 'parietal'
            # img = brain.save_montage("/Users/nwilming/Desktop/t.png", )
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
            1.217,
            None,
            colors,
            "Pair",
            auc_cutoff=0.5,
            ax=ax,
            horizontal=True,
            plot_labels=False,
            color_by_cmap=True,
        )
        ax.set_xlabel("ROI")
        ax.set_ylabel("AUC")
        ax.set_title("")
        ax.set_ylim([0.49, 0.7])
        y = np.linspace(0.49, 0.7, 250)
        x = y * 0 - 3
        sns.despine(ax=ax, bottom=True)
        o = corrs(df, t=t).stack().reset_index()
        o.columns = ["idx", "Comparison", "Correlation"]
        #o.Comparison.replace({'Ch./Un.':r'\textcolor{red}{Ch./Un.}'}, inplace=True)
        
        ax = plt.subplot(gs[3, -1])
        sns.stripplot(
            x="Comparison", y="Correlation", color='k', alpha=0.75, dodge=True, jitter=True, ax=ax, data=o
        )
        #ax.set_xticklabels([r'\em{Text to colour}', r'Si', r'un'])
        ax.set_xlabel("")
        sns.despine(ax=ax, bottom=True)
    tight_layout()
    return data


def _figure4B(df=None, gs=None):
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
        "response": (0.5, 0.65),
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
            idx = np.argsort(X.loc[:, 1.2])
            sorting = X.index.values[idx]
            norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
            cm = matplotlib.cm.get_cmap(colormaps[signal])

            for j, roi in enumerate(sorting):
                val, color = X.loc[roi, 1.2], cm(norm(X.loc[roi, 1.2]))

                plt.plot(X.columns.values, X.loc[roi, :], color=color)

                if X.loc[roi, 1.4] > 0.525:
                    R = roi.replace("JWG_", "").replace("dlpfc_", "")
                    # R = re.sub(r'dlpfc.*', 'DLPFC', R)
                    txts.append(plt.text(1.4, X.loc[roi, 1.4] - 0.005, R))
            y = np.linspace(0.475, 0.7, 200)
            x = y * 0 - 0.225
            plt.scatter(x, y, c=cm(norm(y)), marker=0)
            plt.title(titles[signal])
            plt.xlim([-0.25, 1.4])
            plt.ylim([0.475, 0.7])
            plt.xlabel("Time")
            plt.axvline(1.2, color="k", alpha=0.9)
            if i > 0:
                sns.despine(ax=plt.gca(), left=True)
                plt.yticks([])
            else:
                plt.ylabel("AUC")
                sns.despine(ax=plt.gca())

            palette = {
                d.replace("dlpfc_", "").replace("pgACC_", ""): X.loc[d, 1.2]
                for d in X.index.values
            }

            img = _get_lbl_annot_img(
                palette,
                low=low,
                high=high,
                views=[["par", "front"], ["med", "lat"]],
                colormap=colormaps[signal],
            )
            # brain = dp.plot_brain_color_annotations(palette, low=0.3, high=0.7)
            # 'lateral', 'medial', 'rostral',
            # 'caudal', 'dorsal', 'ventral', 'frontal', 'parietal'
            # img = brain.save_montage("/Users/nwilming/Desktop/t.png", )
            plt.subplot(gs[1, i], aspect="equal", zorder=-10)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            sns.despine(ax=plt.gca(), left=True, bottom=True)

    # plt.savefig(
    #    "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_figure3.pdf",
    #    dpi=1200,
    # )
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
    gs = matplotlib.gridspec.GridSpec(9, 3, 
        height_ratios=[.8, 0.4, 0.5, 0.25, 0.8, 0.35, 0.5, 0.45, 0.8], hspace=0.0)
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
        _figure6A(gs=gs[4,:])
        _figure6B(gs=gs[6,:])
        #_figure5C(ssd.test_slope.Averaged, oglssd.test_slope.Pair, gs=gs[3,:])
        savefig("/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/5_figure_5.pdf",
            dpi=1200, bbox_inches='tight')

        plt.figure(figsize=(7.5, 1.5))
        gs = matplotlib.gridspec.GridSpec(1, 3, hspace=0.0)
        _figure5B(gs=gs[:,:])
        savefig("/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_5_figure_5.pdf",
            dpi=1200, bbox_inches='tight')
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
                    "/Users/nwilming/u/conf_analysis/results/all_areas_Xarea_stim_latency.pickle",
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
        gs = matplotlib.gridspec.GridSpec(1, 5, width_ratios=[1, 0.25, 1, 1, 1],)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=gs, width_ratios=[1, 0.35, 1, 1, 1],
        )
    area_names = ['aIPS', 'IPS/PostCeS', 'M1 (hand)']
    area_colors = _stream_palette()
    for i, signal in enumerate(["JWG_aIPS",  "JWG_IPS_PCeS", "JWG_M1", ]):
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
        yshuff = pd.pivot_table(
            columns="latency",
            index="subject",
            values="corr",
            data=sd.query('comparison=="shuff_corr"'),
        )
        t, p = ttest_rel(yc, y1smp)
        t, ps = ttest_rel(yc, yshuff)
        t, pvsnull = ttest_1samp(yc, 0)
        ax = subplot(gs[0, i+2])

        ax.plot(
            yc.columns.values,
            (yc.values).mean(0),
            "-",
            color=sns.xkcd_rgb[colors[0]],
            label='Ten samples',
        )
        ax.set_ylim([-0.01, 0.05])
        ax.plot(
            y1smp.columns.values,
            (y1smp.values).mean(0),
            "-",
            color=sns.xkcd_rgb[colors[1]],
            label='Last sample',
        )
        
        id_cor, _ = fdr_correction(p)  # <0.05
        id_unc = pvsnull<0.05        
        draw_sig(ax, yc-y1smp, fdr=True, lw=2)
        #draw_sig(ax, yc-y1smp, fdr=False, color='g', zorder=0, lw=2)
        #draw_sig(ax, yc, fdr=False, alpha=0.5, lw=1)
        title(area_names[i], fontsize=8, color=area_colors[signal])
        if i == 0:
            ax.set_xlabel('Time after sample onset')
        if i == 0:
            ax.set_ylabel('Correlation')
            ax.legend(frameon=False, loc=9)
            sns.despine(ax=ax)
        else:
            
            ax.set_yticks([])
            sns.despine(ax=ax, left=True)
    ax = subplot(gs[0, 0])
    #img = _get_lbl_annot_img({'vfcPrimary':0.31, 'JWG_M1':1, 'JWG_aIPS':0.8, 'JWG_IPS_PCeS':0.9}, 
    #    low=0.1, high=1, views=[["par"]], colormap='viridis')
    img = _get_palette(
        {'vfcPrimary':area_colors['vfcPrimary'], 
         'JWG_M1':area_colors['JWG_M1'], 
         'JWG_aIPS':area_colors['JWG_aIPS'], 
         'JWG_IPS_PCeS':area_colors['JWG_IPS_PCeS']}, 
         views=[["par"]])
    ax.imshow(img, aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    from matplotlib import patches
    style="Simple,tail_width=0.5,head_width=4,head_length=8"
    
    a3 = patches.FancyArrowPatch((501, 559), (295,151),
        connectionstyle="arc3,rad=.25",**dict(arrowstyle=style, color=area_colors['JWG_aIPS']))
    a2 = patches.FancyArrowPatch((501, 559), (190,229),
        connectionstyle="arc3,rad=.15",**dict(arrowstyle=style, color=area_colors['JWG_IPS_PCeS']))
    a1 = patches.FancyArrowPatch((501, 559), (225,84),
        connectionstyle="arc3,rad=.55",**dict(arrowstyle=style, color=area_colors['JWG_M1']))
    ax.add_patch(a3)
    ax.add_patch(a2)
    ax.add_patch(a1)
    sns.despine(ax=ax, left=True, bottom=True)


def _figure5C(ssd, oglssd, gs=None):
    if gs is None:
        gs = matplotlib.gridspec.GridSpec(1, 4)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=gs, width_ratios=[1, 1, .5, .5],
        )
    from conf_analysis.behavior import kernels, empirical
    from scipy.stats import ttest_1samp
    from mne.stats import fdr_correction
    import pickle
    #dz = empirical.get_dz()
    data = empirical.load_data()        
    K = data.groupby('snum').apply(kernels.get_decision_kernel)
    #K = dp.extract_kernels(data, contrast_mean=0.5, include_ref=True).T    
    C = data.groupby('snum').apply(kernels.get_confidence_kernel).stack().groupby('snum').mean()    
    
    ax = subplot(gs[0, -1])
    ax.plot(K.mean(0), color='k', label='Choice') #-K.mean(0).mean()
    ax.plot(C.mean(0), color=(0.5, 0.5, 0.5), label='Confidence') #-C.mean(0).mean()
    ax.legend(frameon=False, bbox_to_anchor= (0.3, 1), loc='upper left')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Behavior')
    ax.set_yticks([])
    ax.set_xticks([])
    #ax.set_title('behavior')
    #ax.set_xticks(np.arange(10))
    #ax.set_xticklabels({k+1:k+1 for k in np.arange(10)})
    sns.despine(ax=ax, left=True)
    ax = subplot(gs[0, -2])
    ex_kernels = pickle.load(open('/Users/nwilming/u/conf_analysis/results/example_kernel_vfcPrimary.pickle', 'rb'))
    ex_choice = ex_kernels['choice']
    ax.plot((ex_choice.mean(0)), color='k', label='Choice') #-ex_choice.mean()
    ex_conf = ex_kernels['conf']
    ax.plot((ex_conf.mean(0)), color=(0.5, 0.5, 0.5), label='Confidence') #ex_conf.mean()
    #ax.legend(frameon=False, bbox_to_anchor= (0.3, 1), loc='upper left')
    #ax.set_title('V1')
    ax.set_xlabel('Sample')
    ax.set_ylabel('V1')
    ax.set_yticks([])
    ax.set_xticks([])
    #ax.set_xticks(np.arange(10))
    #ax.set_xticklabels({k+1:k+1 for k in np.arange(10)})
    sns.despine(ax=ax, left=True)

    cck = pd.read_hdf('/Users/nwilming/u/conf_analysis/results/choice_kernel_correlations.hdf')
    cm = cck.groupby('cluster').mean()
    K_t_palette = dict(cm.choice_corr)
    C_t_palette = dict(cm.conf_corr)
    ax = subplot(gs[0, 0])
    voffset=1
    low = -0.6
    high = 0.6
    img = _get_lbl_annot_img({k:v+voffset for k,v in K_t_palette.items()}, views=["lat", "med"], low=voffset+low, high=voffset+high, thresh=None)
    ax.imshow(img, aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Choice kernel correlation')
    sns.despine(ax=ax, left=True, bottom=True)
    
    ax = subplot(gs[0, 1])
    img = _get_lbl_annot_img({k:v+voffset for k,v in C_t_palette.items()}, views=["lat", "med"], low=voffset+low, high=voffset+high, thresh=None)
    ax.imshow(img, aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Corr. w Conf. kernel')
    sns.despine(ax=ax, left=True, bottom=True)
    return #K_p_palette, K_t_palette    


figure_6colors = sns.color_palette(n_colors=3)
figure_6colors = {'AFcorr':figure_6colors[0], 'DCD':figure_6colors[1], 'AF-CIFcorr':figure_6colors[2],
        'lAFcorr':figure_6colors[0], 'lAF-CIFcorr':figure_6colors[2]}

def _figure6B(cluster='vfcPrimary', gs=None):
    import pickle
    from scipy.stats import linregress, ttest_1samp
    res = []
    for freq in range(10, 115, 5):
        fname = '/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle'%freq
        a = pickle.load(open(fname,'rb'))
        ccs, K, kernels, v1decoding = a['ccs'], a['K'], a['kernels'], a['v1decoding']
        kernels.set_index(['cluster', 'rmcif', 'subject'], inplace=True)
        KK = np.stack(kernels.kernel)
        kernels = pd.DataFrame(KK, index=kernels.index).query('cluster=="%s"'%cluster)
        rems = pd.pivot_table(data=kernels.query('rmcif==True'), index='subject')
        alls = pd.pivot_table(data=kernels.query('rmcif==False'), index='subject')
        diff = alls.mean(1).to_frame()        
        diff.loc[:, 'comparison'] = 'V1kernel'
        diff.loc[:, 'freq'] = freq
        diff.columns = ['sum', 'comparison', 'freq']
        res.append(diff)
        diff = rems.mean(1).to_frame()        
        diff.loc[:, 'comparison'] = 'V1kernelCIF'
        diff.loc[:, 'freq'] = freq
        diff.columns = ['sum', 'comparison', 'freq']
        res.append(diff)
        alls = alls.apply(lambda x:linregress(np.arange(10), x)[0], axis=1).to_frame()
        #alls.mean(1).to_frame()        
        alls.loc[:, 'comparison'] = 'slope'
        alls.loc[:, 'freq'] = freq
        alls.columns = ['sum', 'comparison', 'freq']
        res.append(alls)
    res = pd.concat(res)
    if gs is None:
        gs = matplotlib.gridspec.GridSpec(1, 2)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs, width_ratios=[1,0.125, 1]
        )
    ax = plt.subplot(gs[0,0])
    allv1 = pd.pivot_table(data=res.query('comparison=="V1kernel"'),
        index='subject', columns='freq', values='sum')
    plotwerr(allv1, color=figure_6colors['AFcorr'])
    
    cifrem = pd.pivot_table(data=res.query('comparison=="V1kernelCIF"'),
        index='subject', columns='freq', values='sum')
    plotwerr(cifrem, color=figure_6colors['AF-CIFcorr'])
    #draw_sig(ax,diff, fdr=False, color='k')
    draw_sig(ax, allv1-cifrem, fdr=True, color='k')

    xlabel('Frequency')
    ylabel('Kernel sum')
    yticks([-0.005, 0,  0.005, 0.01])
    sns.despine(ax=ax)
    ax = plt.subplot(gs[0,-1])
    diff = pd.pivot_table(data=res.query('comparison=="slope"'),
        index='subject', columns='freq', values='sum')

    plotwerr(diff)
    p = ttest_1samp(diff.values, 0)[0]
    #draw_sig(ax,diff, color='k')
    draw_sig(ax,diff, fdr=False, color='k')
    #draw_sig(ax,diff, fdr=True, color='g')
    xlabel('Frequency')
    ylabel('Slope of\nV1 kernel')
    yticks([-0.002, 0, 0.002])
    sns.despine(ax=ax)
    return res


def _figure6A(cluster='vfcPrimary', freq='gamma', lowfreq=10, gs=None, 
    label=True):
    import matplotlib
    import seaborn as sns
    import pickle

    fs = {
        'gamma':'/Users/nwilming/u/conf_analysis/results/cort_kernel.results.pickle', 
        'alpha':'/Users/nwilming/u/conf_analysis/results/cort_kernel_f0-10.results.pickle',
        'beta':'/Users/nwilming/u/conf_analysis/results/cort_kernel_f13-30.results.pickle',        
    }
    try:
        a = pickle.load(open(fs[freq],'rb'))
    except KeyError:
        fname = '/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle'%freq
        a = pickle.load(open(fname,'rb'))
    ccs, K, kernels, peaks = a['ccs'], a['K'], a['kernels'], a['v1decoding']
    v1decoding = peaks['V1']    

    fname = '/Users/nwilming/u/conf_analysis/results/cort_kernel_f%i.results.pickle'%lowfreq
    b = pickle.load(open(fname,'rb'))
    low_kernels = b['kernels']

    v1decoding = peaks['V1']    
    colors = figure_6colors
    
    #plt.figure(figsize=(12.5, 5.5))
    if gs is None:
        figure(figsize=(10.5, 3))
        gs = matplotlib.gridspec.GridSpec(1, 8,
            width_ratios=[1, 0.5, 0.25, 1, 1, 0.25,  0.5, 0.5])
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 8, subplot_spec=gs, width_ratios=[1, 0.75, 0.25, 1, 1, 0.55, 0.5, 0.5]
        )
    #gs = matplotlib.gridspec.GridSpec(1, 4)
    ax = plt.subplot(gs[0, 0])
    kernels.set_index(['cluster', 'rmcif', 'subject'], inplace=True)
    KK = np.stack(kernels.kernel)
    kernels = pd.DataFrame(KK, index=kernels.index).query('cluster=="%s"'%cluster)
    rems = pd.pivot_table(data=kernels.query('rmcif==True'), index='subject')
    alls = pd.pivot_table(data=kernels.query('rmcif==False'), index='subject')
    

    low_kernels.set_index(['cluster', 'rmcif', 'subject'], inplace=True)
    low_KK = np.stack(low_kernels.kernel)
    low_kernels = pd.DataFrame(low_KK, index=low_kernels.index).query('cluster=="%s"'%cluster)
    low_rems = pd.pivot_table(data=low_kernels.query('rmcif==True'), index='subject')
    low_alls = pd.pivot_table(data=low_kernels.query('rmcif==False'), index='subject')

    plotwerr(K, color='k', label='Behavioral kernel', lw=1)
    plotwerr(v1decoding.T,  label='V1 contrast', color=colors['DCD'],
     lw=1)

    plt.yticks([0, 0.04, 0.08, 0.12])
    plt.xticks(np.arange(10), ['1'] + ['']*8 + ['10'])
    plt.xlabel('Sample')
    plt.ylabel('AUC / Encoding strength')
    #plt.axhline(0, color='k', lw=1)
    plt.legend(frameon=False, ncol=1, loc='center left', 
        bbox_to_anchor=[0.2, 1], fontsize=8)
    sns.despine(ax=ax, bottom=False, right=True)

    ax = plt.subplot(gs[0, -5], zorder=1)
    plotwerr(alls, label='V1 kernel', color=colors['AFcorr'])
    plotwerr(rems, label='Contrast fluctu-\nations removed', color=colors['AF-CIFcorr'])    
    #plt.xticks(np.arange(10), np.arange(10)+1)
    plt.xticks(np.arange(10), ['1'] + ['']*8 + ['10'])
    plt.xlabel('Sample')
    plt.ylabel('AUC')
    plt.yticks([0, 0.02, 0.04])
    yl = plt.ylim()
    #plt.axhline(0, color='k', lw=1)
    plt.legend(frameon=False, ncol=1, loc='center left', 
        bbox_to_anchor=[0.15, .8], fontsize=8)
    sns.despine(ax=ax, bottom=False, right=True)
    plt.title('Gamma', fontsize=8)
    ax = plt.subplot(gs[0, -4], zorder=0)

    plotwerr(low_alls,  ls=':',  color=colors['AFcorr']) #label='V1 kernel',
    plotwerr(low_rems,  ls=':',  color=colors['AF-CIFcorr']) #label='Contrast fluctu-\nations removed',
    #plt.xticks(np.arange(10), np.arange(10)+1)
    plt.xticks(np.arange(10), ['1'] + ['']*8 + ['10'])
    plt.xlabel('Sample')
    plt.ylim(yl)
    plt.ylabel(None)
    plt.yticks([])
    plt.title('Alpha', fontsize=8)
    #plt.axhline(0, color='k', lw=1)

    sns.despine(ax=ax, bottom=False, right=True, left=True)

    rescorr = []
    for i in range(15):
        rescorr.append(
            {'AFcorr': np.corrcoef(alls.loc[i+1], K.loc[i+1,:])[0,1],
             'AF-CIFcorr':np.corrcoef(rems.loc[i+1], K.loc[i+1,:])[0,1],
             'DCD':np.corrcoef(v1decoding.T.loc[i+1], K.loc[i+1,:])[0,1],
             'DCD-AFcorr':np.corrcoef(v1decoding.T.loc[i+1], alls.loc[i+1,:])[0,1],
             'lAFcorr': np.corrcoef(low_alls.loc[i+1], K.loc[i+1,:])[0,1],
             'lAF-CIFcorr':np.corrcoef(low_rems.loc[i+1], K.loc[i+1,:])[0,1],
            'subject':i+1
            }
        )
    rescorr = pd.DataFrame(rescorr)
    rs = rescorr.set_index('subject').stack().reset_index()
    rs.columns = ['subject', 'comparison', 'correlation']

    ax = plt.subplot(gs[0, -2])
    sns.stripplot(data=rs, x='comparison', y='correlation', 
        order=['AFcorr', 'AF-CIFcorr'],
        palette=colors)

    plt.ylabel('Correlation with\nbehavioral kernel')
    plt.plot([-0.125, +0.125], [rescorr.loc[:, 'AFcorr'].mean(), rescorr.loc[:, 'AFcorr'].mean()], 'k', zorder=100)        
    plt.plot([1-0.125, 1+0.125], [rescorr.loc[:, 'AF-CIFcorr'].mean(), rescorr.loc[:, 'AF-CIFcorr'].mean()], 'k', zorder=100)
    plt.title('Gamma', fontsize=8)
    plt.xlabel('')
    plt.xticks([])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.ylim([-1, 1])
    yl = plt.ylim()
    sns.despine(ax=ax, bottom=True)

    ax = plt.subplot(gs[0, -1])
    sns.stripplot(data=rs, x='comparison', y='correlation', 
        order=['lAFcorr', 'lAF-CIFcorr'],
        palette=colors)
    plt.ylim(yl)
    plt.plot([-0.125, 0+0.125], [rescorr.loc[:, 'lAFcorr'].mean(), rescorr.loc[:, 'lAFcorr'].mean()], 'k', zorder=100)        
    plt.plot([1-0.125, 1+0.125], [rescorr.loc[:, 'lAF-CIFcorr'].mean(), rescorr.loc[:, 'lAF-CIFcorr'].mean()], 'k', zorder=100)
    plt.title('Alpha', fontsize=8)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel('')
    plt.xlabel('')
    if label:
        #plt.xticks([0, 1, 2], ['V1 kernel', 'V1 contrast\ndecoding', 'Contrast\nfluctations\nremoved'], rotation=45)
        plt.xticks([])
        
    else:
        plt.xticks([0, 1, 2], ['','',''], rotation=45)
    #plt.plot([3-0.125, 3+0.125], [rescorr.loc[:, 'DCD-AFcorr'].mean(), rescorr.loc[:, 'DCD-AFcorr'].mean()], 'k')
    #plt.xlim([-0.5, 2.5])
    plt.xlabel('')
    sns.despine(ax=ax, bottom=True, left=True)

    ax = plt.subplot(gs[0, 1], zorder=-1)
    pal = {}
    for cluster in peaks.columns.get_level_values('cluster').unique():
        r = []
        for i in range(15):
            r.append(
             np.corrcoef(peaks[cluster].T.loc[i+1], K.loc[i+1,:])[0,1]
            )
        pal[cluster] = np.mean(r)
        
    #pal = dict(ccs.groupby('cluster').mean().loc[:, 'AFcorr'])
    img = _get_lbl_annot_img({k:v+1 for k,v in pal.items()}, low=1+-1, high=1+1, thresh=-1)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)
    

def plot_per_sample_resp(ax, ssd, area, area_label, integration_slice, ylim=None):
    k = dp.sps_2lineartime(dp.get_ssd_per_sample(ssd, "SSD", area=area))
    ka = dp.sps_2lineartime(dp.get_ssd_per_sample(ssd, "SSD_acc_contrast", area=area))
    plt.plot(k.index, k.max(1), color=figure_6colors['DCD'], label="Contrast")
    plt.plot(k.index, k, color=figure_6colors['DCD'], lw=0.1)
    plt.plot(ka.index, ka.max(1), "m", label="Accumulated\ncontrast")
    plt.plot(ka.index, ka, "m", lw=0.1)
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
def _get_palette(palette, brain=None, ogl=False, views=['par', 'med']):
    brain = dp.plot_brain_color_legend(palette, brain=brain, ogl=ogl, subject="fsaverage")
    return brain.save_montage('/Users/nwilming/Desktop/t.png', views)


@memory.cache()
def _get_lbl_annot_img(
    palette, low=0.4, high=0.6, views=[["lat"], ["med"]], colormap="RdBu_r", thresh=0,
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


def make_stimulus(contrast, baseline=0.5):
    contrast = (contrast/2);
    low = baseline-contrast
    high = baseline+contrast
    shift=1 
    ppd = 45
    sigma = 75
    cutoff = 5.5
    radius = 4*ppd
    ringwidth = ppd*3/4
    inner_annulus = 1.5*ppd
    X, Y = np.meshgrid(np.arange(1980), np.arange(1024))    

    #/* Compute euclidean distance to center of our ring stim: */
    #float d = distance(pos, RC);
    d = ((X-1980/2)**2 + (Y-1024/2)**2)**.5

    #/* If distance greater than maximum radius, discard this pixel: */
    # if (d > Radius + Cutoff * Sigma) discard; 
    # if (d < Radius - Cutoff * Sigma) discard; 
    # if (d < Annulus) discard;

    #float alpha = exp(-pow(d-Radius,2.)/pow(2.*Sigma,2.));
    alpha = np.exp(-(d-radius)**2/(2*sigma)**2)

    #/* Convert distance from units of pixels into units of ringwidths, apply shift offset: */
    #d = 0.5 * (1.0 + sin((d - Shift) / RingWidth * twopi));
    
    rws = 0.5 * (1.0 + np.sin((d - shift) / ringwidth * 2*np.pi))

    #/* Mix the two colors stored in gl_Color and secondColor, using the slow
    # * sine-wave weight term in d as a mix weight between 0.0 and 1.0:
    # */
    #gl_FragColor = ((mix(firstColor, secondColor, d)-0.5) * alpha) + 0.5;
    rws = high*(1-rws)+low*rws
    rws[d>(radius+cutoff*sigma)] = 0.5
    rws[d<(radius-cutoff*sigma)] = 0.5
    rws[d<inner_annulus] = 0.5

    return rws