"""
Make figures for manuscript.
"""
import pandas as pd
import matplotlib
from pylab import *
from conf_analysis.behavior import metadata
from conf_analysis.meg import decoding_plots as dp
from joblib import Memory
memory = Memory(location=metadata.cachedir, verbose=-1)

def_ig = slice(0.4, 1.15) 


def figure1(df=None, stats=False):
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

    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 0.5}):
        figure(figsize=(7.5, 7.5))
        from conf_analysis.meg import srtfr

        gs = matplotlib.gridspec.GridSpec(3, 2, width_ratios=[0.99, 0.01])

        srtfr.plot_stream_figures(
            df.query('hemi=="avg"'),
            contrasts=["all"],
            flip_cbar=False,
            gs=gs[0, 0],
            stats=stats,
        )

        gs0 = matplotlib.gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs[0, 1])
        subplot(gs0[1:5, 0])
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-50, 50), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        colorbar(cmap, cax=gca(), shrink=0.5, ticks=[-50, 0, 50], drawedges=False)

        srtfr.plot_stream_figures(
            df.query('hemi=="avg"'),
            contrasts=["stimulus"],
            flip_cbar=False,
            gs=gs[1, 0],
            stats=stats,
        )

        gs0 = matplotlib.gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs[1, 1])
        subplot(gs0[1:5, 0])
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-25, 25), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        colorbar(cmap, cax=gca(), shrink=0.5, ticks=[-25, 0, 25], drawedges=False)

        srtfr.plot_stream_figures(
            df.query('~(hemi=="avg")'),
            contrasts=["choice"],
            flip_cbar=True,
            gs=gs[2, 0],
            stats=stats,
        )

        gs0 = matplotlib.gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs[2, 1])
        subplot(gs0[1:5, 0])
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(-25, 25), cmap=plt.get_cmap("RdBu_r")
        )
        cmap.set_array([])
        colorbar(cmap, cax=gca(), shrink=0.5, ticks=[-25, 0, 25], drawedges=False)
        tight_layout()
        savefig(
            "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/fig1.pdf"
        )
    return df, stats


def figure2(df=None):
    """
    Plot Decoding results
    """
    if df is None:
        df = dp.get_decoding_data()
    palette = dp.get_area_palette()
    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 0.5}):
        plt.figure(figsize=(8, 5.5))
        plotter = dp.StreamPlotter(
            dp.plot_config,
            {"MIDC_split": "Reds", "CONF_unsigned": "Blues", "CONF_signed": "Greens"},
            {
                "Averaged": df.test_roc_auc.Averaged,
                "Lateralized": df.test_roc_auc.Lateralized,
            },
        )
        plotter.plot()

        plt.tight_layout()
    try:
        savefig(
            "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/fig2.pdf"
        )
    except FileNotFoundError:
        savefig(
            "/home/student/n/nwilming/conf_fig2.pdf"
        )
    return df


def figure3(df=None, quick=False):
    """
    Reproduce state space plot and decoding of all areas at
    t = peak sensory decoding
    """
    import pymc3 as pm
    import seaborn as sns

    colors = {"MIDC_split": "r", "CONF_unsigned": "b", "CONF_signed": "g"}
    figure(figsize=(8, 4.5))
    if df is None:
        df = dp.get_decoding_data()
    gs = matplotlib.gridspec.GridSpec(2, 4, hspace=0.5, wspace=0.3)
    subplot(gs[:, :2])
    if not quick:
        dp.state_space_plot(
            df.test_roc_auc.Lateralized,
            "MIDC_split",
            "CONF_unsigned",
            df_b=df.test_roc_auc.Lateralized,
            a_max=0.55,
            color=sns.xkcd_rgb["purple"],
        )
        dp.state_space_plot(
            df.test_roc_auc.Lateralized,
            "MIDC_split",
            "CONF_signed",
            df_b=df.test_roc_auc.Lateralized,
            a_max=0.55,
            color=sns.xkcd_rgb["orange"],
        )
        dp.state_space_plot(
            df.test_roc_auc.Averaged,
            "MIDC_split",
            "CONF_signed",
            df_b=df.test_roc_auc.Averaged,
            a_max=0.55,
            color="k",
        )
    plt.ylim([0.45, 0.75])
    plt.xlim([0.45, 0.75])
    plt.xlabel("AUC")
    plt.ylabel("AUC")
    sns.despine(ax=plt.gca())

    plt.plot([0.5, 0.75], [0.5, 0.75], "k", lw=0.5)
    sorting, ax_diff_stim = dp.plot_signal_comp(
        df,
        latency=1.2,
        auc_cutoff=0.55,
        gs=gs[0, 2:],
        xlim=[0.45, 0.9],
        xlim_avg=[0.49, 0.65],
        colors=colors,
    )

    # Get diff between absolute conf between averaged H and lateralized H
    signals, subjects, areas, dataA = dp.get_signal_comp_data(
        df.test_roc_auc.Averaged, 1.2, "stimulus"
    )
    signals, subjects, areas, dataL = dp.get_signal_comp_data(
        df.test_roc_auc.Lateralized, 1.2, "stimulus"
    )

    from conf_analysis.meg import stats

    k, mdl = stats.auc_get_sig_cluster_group_diff_posterior(dataA, dataL)
    hpd = pm.stats.hpd(k.get_values("mu_diff"))

    for j, signal in enumerate(signals):
        mean = (dataL[j, :, :] - dataA[j, :, :]).mean(-1).squeeze()
        for i, s in enumerate(sorting):
            ax_diff_stim.plot(mean[s], i, ".", color=colors[signal])
            ax_diff_stim.plot(hpd[s, j, :], [i, i], color=colors[signal])

    sns.despine(left=True, ax=ax_diff_stim)
    ax_diff_stim.set_xlim([-0.2, 0.2])
    ax_diff_stim.set_xticks([-0.1, 0, 0.1], minor=False)
    ax_diff_stim.set_xticklabels(["-0.1", "", "0.1"])
    ax_diff_stim.axvline(0, color="k", lw=0.5)
    ax_diff_stim.set_yticks([])

    # legend()
    sorting, ax_diff_stim = dp.plot_signal_comp(
        df,
        latency=0,
        auc_cutoff=0.55,
        gs=gs[1, 2:],
        epoch="response",
        xlim=[0.45, 0.9],
        xlim_avg=[0.49, 0.65],
        idx=sorting,
        colors=colors,
    )

    # Get diff between absolute conf between averaged H and lateralized H
    signals, subjects, areas, dataA = dp.get_signal_comp_data(
        df.test_roc_auc.Averaged, 0, "response"
    )
    signals, subjects, areas, dataL = dp.get_signal_comp_data(
        df.test_roc_auc.Lateralized, 0, "response"
    )

    from conf_analysis.meg import stats

    k, mdl = stats.auc_get_sig_cluster_group_diff_posterior(dataA, dataL)
    hpd = pm.stats.hpd(k.get_values("mu_diff"))
    for j, signal in enumerate(signals):
        mean = (dataL[j, :, :] - dataA[j, :, :]).mean(-1).squeeze()
        for i, s in enumerate(sorting):
            ax_diff_stim.plot(mean[s], i, ".", color=colors[signal])
            ax_diff_stim.plot(hpd[s, j, :], [i, i], color=colors[signal])
    ax_diff_stim.set_xlim([-0.2, 0.2])
    ax_diff_stim.set_xticks([-0.1, 0, 0.1], minor=False)
    ax_diff_stim.set_xticklabels(["-0.1", "", "0.1"])
    ax_diff_stim.axvline(0, color="k", lw=0.5)
    ax_diff_stim.set_yticks([])
    sns.despine(left=True, ax=ax_diff_stim)
    savefig(
        "/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/fig3.pdf",
        bbox_inches="tight",
    )
    return k, mdl




def figure4(ssd=None, idx=None, brain=None, integration_slice=def_ig):
    if ssd is None:
        ssd = dp.get_ssd_data(restrict=False)
    if idx is None:
        idx = dp.get_ssd_idx(ssd.test_slope, integration_slice=integration_slice)
    if "post_medial_frontal" in ssd.test_slope.columns:
        del ssd.test_slope["post_medial_frontal"]
    if "vent_medial_frontal" in ssd.test_slope.columns:
        del ssd.test_slope["vent_medial_frontal"]
    if "ant_medial_frontal" in ssd.test_slope.columns:
        del ssd_test_slope["ant_medial_frontal"]
    palette, brain = dp.ssd_index_plot(
        idx,
        ssd.test_slope,
        labels=None,
        rgb=True,
        brain=brain,
        integration_slice=integration_slice,
    )
    savefig("/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/fig4.pdf")
    return ssd, idx, brain


def sup_figure1(df=None):
    import seaborn as sns
    if df is None:
        df = dp.get_decoding_data(restrict=False, ogl=True)
    stim = df.test_roc_auc.Pair.query('epoch=="stimulus"')
    resp = df.test_roc_auc.Pair.query('epoch=="response"')
    
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=0.3, vmax=0.7)
    cm = matplotlib.cm.get_cmap("RdBu_r")
    cnt=0
    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 0.5}):
        plt.figure(figsize=(9, 3))
        gs = matplotlib.gridspec.GridSpec(3, 13)       
        
        for j, signal in enumerate(["MIDC_split", "CONF_signed", "CONF_unsigned"]):
            # First plot Stimulus locked
            latencies = stim.index.get_level_values("latency").unique() 
            stim_latencies = np.arange(0.3, 1.3, 0.1)
            for i, latency in enumerate(stim_latencies):        
                target_lat = latencies[np.argmin(np.abs(latencies - latency))]
                d = (
                    stim.query('latency==%f & signal=="%s"' % (target_lat, signal))
                    .groupby("signal")
                    .mean()
                )                
                palette =  {k:d[k].values[0] for k in d.columns}             
                img = _get_img(palette, low=0.3, high=0.7)       
                plt.subplot(gs[j, i])
                plt.imshow(img, aspect='equal')
                plt.xticks([])
                plt.yticks([])
                sns.despine(left=True, bottom=True)
                if signal=="MIDC_split":
                    n = ('%0.1f'%target_lat).replace('0','')                
                    title(r't=%s'%n)
                if i == 0:
                    ylabel(signal)                
            # Now response locked
            latencies = resp.index.get_level_values("latency").unique()
            for i, latency in enumerate(np.arange(-0.2, 0.01, 0.1)):        
                target_lat = latencies[np.argmin(np.abs(latencies - latency))]
                d = (
                    resp.query('latency==%f & signal=="%s"' % (target_lat, signal))
                    .groupby("signal")
                    .mean()
                )
                palette =  {k:d[k].values[0] for k in d.columns}             
                img = _get_img(palette, low=0.3, high=0.7)       
                plt.subplot(gs[j, i+len(stim_latencies)])
                plt.imshow(img, aspect='equal')
                plt.xticks([])
                plt.yticks([])
                sns.despine(left=True, bottom=True)
                if signal=="MIDC_split":
                    n = ('%0.1f'%target_lat).replace('0','')
                    if n=='.':
                        n='0'
                    title(r't=%s'%n)
    plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_figure1.pdf', dpi=1200)
    return df


def sup_figure2(ssd=None, idx=None, integration_slice=def_ig):
    import seaborn as sns    
    import matplotlib
    
    if ssd is None:
        ssd = dp.get_ssd_data(restrict=False, ogl=True)
    if idx is None:
        idx = dp.get_ssd_idx(ssd.test_slope, integration_slice=integration_slice, pair=True)
    
    with mpl.rc_context(rc={"font.size": 8.0, "lines.linewidth": 0.5}):
        plt.figure(figsize=(5.5, 4))
        dv = 10
        plt.subplot(1,3,1)
        m = idx.groupby('cluster').mean()
        palette =  {k:dv+vals.SSD for k, vals in m.iterrows()} 
        img = _get_img(palette, low=dv+-0.05, high=dv+0.05)     
        plt.imshow(img, aspect='equal')
        plt.xticks([])
        plt.yticks([])
        plt.title('Contrast enc.')

        plt.subplot(1,3,2)
        m = idx.groupby('cluster').mean()
        palette =  {k:dv+vals.SSD_acc_contrast for k, vals in m.iterrows()} 
        img = _get_img(palette, low=dv+-0.05, high=dv+0.05)     
        plt.imshow(img, aspect='equal')
        plt.xticks([])
        plt.yticks([])
        plt.title('Acc. contrast')
        plt.subplot(1,3,3)
        
        m = idx.groupby('cluster').mean()
        palette =  {k:dv+vals.SSDvsACC for k, vals in m.iterrows()} 
        img = _get_img(palette, low=dv+-0.025, high=dv+0.025)     
        plt.imshow(img, aspect='equal')
        plt.xticks([])
        plt.yticks([])
        plt.title('Sensitivity idx.')
        sns.despine(left=True, right=True, bottom=True)
    plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/manuscript/figures/supp_figure2.pdf', dpi=1200)
    return ssd, idx

@memory.cache()
def _get_img(palette, low=0.3, high=0.7):
    brain = dp.plot_brain_color_annotations(palette, low=low, high=high)
    return brain.save_montage(
            "/Users/nwilming/Desktop/t.png", [["lat"], ["med"]]
    )