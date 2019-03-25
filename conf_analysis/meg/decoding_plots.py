import numpy as np
import os
import pandas as pd
import pylab as plt
import seaborn as sns

from matplotlib import cm
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import pyplot
from pymeg import atlas_glasser as ag
from pymeg import source_reconstruction as sr

from conf_analysis.behavior import metadata

from joblib import Memory

if 'RRZ_LOCAL_TMPDIR' in os.environ.keys():
    memory = Memory(cachedir=os.environ['RRZ_LOCAL_TMPDIR'])
if 'TMPDIR' in os.environ.keys():
    tmpdir = os.environ['TMPDIR']
    memory = Memory(cachedir=tmpdir)
else:
    memory = Memory(cachedir=metadata.cachedir)


choice_decoding_areas = ('HCPMMP1_premotor', 'JWG_M1',
                         'vfcFEF', 'JWG_IPS_PCeS',
                         'HCPMMP1_dlpfc', 'JWG_aIPS')


def filter_latency(data, min, max):
    lat = data.index.get_level_values('latency').values
    return data.loc[(min < lat) & (lat < max), :]


def get_decoding_data(decoding_classifier="SCVlin", restrict=True):
    df = pd.read_hdf(
        '/Users/nwilming/u/conf_analysis/results/all_decoding_results_20190215.hdf')
    df.loc[:, 'latency'] = df.latency.round(3)
    idnan = np.isnan(df.subject)
    df.loc[idnan, 'subject'] = df.loc[idnan, 'sub']
    df = df.loc[~np.isnan(df.subject), :]
    df = df.query(
        'Classifier=="%s"' % decoding_classifier)
    df.loc[:, 'cluster'] = [(c
                             .split(' ')[0]
                             .replace('_LH', '')
                             .replace('_RH', ''))
                            for c in df.loc[:, 'cluster'].values]
    if restrict:
        clusters = ag.areas.values()
        idx = [True if c in clusters else False for c in df.loc[:, 'cluster']]
        df = df.loc[idx, :]
    for field in ['signal', 'hemi', 'cluster', 'Classifier', 'epoch']:
        df.loc[:, field] = df.loc[:, field].astype('category')
    df.loc[:, 'mc<0.5'] = df.loc[:, 'mc<0.5'].astype(str)
    df.set_index(['Classifier', 'signal', 'subject', 'epoch', 'latency',
                  'mc<0.5', 'hemi', 'cluster'], inplace=True)
    df = df.loc[~df.index.duplicated()]
    df = df.unstack(['hemi', 'cluster'])
    idt = df.test_accuracy.index.get_level_values('signal') == 'CONF_signed'
    df.loc[idt, 'test_accuracy'] = (df.loc[idt, 'test_accuracy'] - 0.25).values
    df.loc[~idt, 'test_accuracy'] = (
        df.loc[~idt, 'test_accuracy'] - 0.5).values
    return df


def get_ssd_data(ssd_classifier="Ridge", restrict=True):
    try:
        df = pd.read_hdf(
            '/Users/nwilming/u/conf_analysis/results/all_decoding_ssd_20190129.hdf')
    except FileNotFoundError:
        df = pd.read_hdf(
            '/home/nwilming/conf_analysis/results/all_decoding_ssd_20190129.hdf')
    df = df.loc[~np.isnan(df.subject), :]
    df = df.query(
        'Classifier=="%s"' % ssd_classifier)
    df.loc[:, 'cluster'] = [c.split(' ')[0].replace('_LH', '').replace('_RH', '')
                            for c in df.loc[:, 'cluster'].values]
    if restrict:
        clusters = ag.areas.values()
        idx = [True if c in clusters else False for c in df.loc[:, 'cluster']]
        df = df.loc[idx, :]
    for field in ['signal', 'hemi', 'cluster', 'Classifier', 'epoch']:
        df.loc[:, field] = df.loc[:, field].astype('category')
    df.set_index(['Classifier', 'signal', 'subject', 'epoch', 'latency',
                  'sample', 'hemi', 'cluster'], inplace=True)
    df = df.loc[~df.index.duplicated()]
    df = df.unstack(['hemi', 'cluster'])
    return df


def get_cvals(epoch, data):
    if epoch == 'response':
        latency = 0
    else:
        latency = 1.25
    p = data.query('latency==%f' % latency).groupby('latency').mean().max()
    return {k: v for k, v in p.items()}


@memory.cache
def get_posterior(data):
    '''
    Get uncertainty around average mean by means of bayesian 
    inference. For AUC values, nothing else.
    '''
    import pymc3 as pm
    n_t = data.shape[1]
    with pm.Model():
        mu = pm.Normal('mu', 0.5, 1, shape=n_t)
        std = pm.Uniform('std', lower=0, upper=1, shape=n_t)
        v = pm.Exponential('ν_minus_one', 1 / 29.) + 1
        pm.StudentT('Out', mu=mu, lam=std**-2, nu=v, shape=n_t, observed=data)
        k = pm.sample()
    mu = k.get_values('mu')
    return mu.mean(0), pm.stats.hpd(mu)


def plot_individual_areas_with_stats(data, type='Pair'):
    palette = get_area_palette()
    for cluster in data.columns:
        plot_signals_hand(data.loc[:, cluster].reset_index(), palette,
                          'AUC',
                          midc_ylim=(.1, .9),
                          conf_ylim=(.5, .7),
                          cortex_cmap='RdBu_r',
                          midc_ylim_cortex=(0.1, 0.9),
                          conf_ylim_cortex=(0.3, 0.7),
                          plot_uncertainty=True,
                          suffix='_%s_%s' % (type, cluster),
                          lw=1)


def plot_signals_hand(data, palette, measure,
                      classifier='svm',
                      midc_ylim=(-0.25, 0.25),
                      conf_ylim=(-0.05, 0.25),
                      midc_ylim_cortex=(-0.25, 0.25),
                      conf_ylim_cortex=(-0.05, 0.25),
                      cortex_cmap='RdBu_r', suffix='all',
                      plot_uncertainty=False,
                      **kw):
    '''
    Plot decoding signals.
    This is the place to start!
    '''

    allc, vfc, glasser, jwg = ag.get_clusters()
    col_order = list(glasser.keys()) + \
        list(vfc.keys()) + list(jwg.keys())
    plt.figure(figsize=(8, 6))
    combinations = [
        ('stimulus', 'MIDC_nosplit'), ('stimulus',
                                       'MIDC_split'), ('response', 'MIDC_split'),
        (None, None), ('stimulus', 'CONF_signed'), ('response', 'CONF_signed'),
        (None, None), ('stimulus', 'CONF_unsigned'), ('response', 'CONF_unsigned')]
    index = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
             (1, 2), (2, 0), (2, 1), (2, 2)]
    gs = gridspec.GridSpec(3, 6)

    for i, (epoch, signal) in enumerate(combinations):
        if epoch is None:
            continue
        row, col = index[i]
        plt.subplot(gs[row, col * 2])
        d = data.query('epoch=="%s" & signal=="%s"' %
                       (epoch, signal))

        cvals = {}
        d = d.groupby(['subject', 'latency', 'mc<0.5']).mean()
        for split, ds in d.groupby('mc<0.5'):
            for column in col_order:
                try:
                    values = pd.pivot_table(data=ds,
                                            index='subject', columns='latency',
                                            values=column)
                    latency = values.columns.values
                    if split == "True":
                        values = -values + 1

                    if plot_uncertainty:
                        mu, hdi = get_posterior(values.values)
                        plt.plot(latency, latency * 0 + 0.5, 'k', zorder=-1)
                        plt.plot(latency,
                                 mu,
                                 color='k',
                                 **kw)
                        plt.fill_between(
                            latency, hdi[:, 0], hdi[:, 1],
                            color=palette[column],
                            alpha=0.75)
                    else:
                        plt.plot(latency,
                                 values.values.mean(0),
                                 color=palette[column],
                                 **kw)
                except KeyError:
                    pass

        cylim = []
        if 'MIDC' in signal:
            plt.ylim(midc_ylim)
            cylim = midc_ylim_cortex
            # pyplot.locator_params(nticks=5)
        else:
            plt.ylim(conf_ylim)
            cylim = midc_ylim_cortex
            # pyplot.locator_params(nticks=5)

        if (col == 0) or ((col == 1) and (row > 0)):
            plt.ylabel('%s\n\n' % signal + r'$%s$' % measure)
        if (row == 2) or ((row == 0) and (col == 0)):
            plt.xlabel(r'$time$')
        center = (plt.xlim()[1] + plt.xlim()[0]) / 2.
        #plt.text(plt.xlim()[0] + 0.05, 0.18, signal, size=8)

        sns.despine(ax=plt.gca(), trim=False)
        if epoch == 'response':
            plt.axvline(0, color='k', zorder=-1, alpha=0.9)
        else:
            plt.axvline(1.25, color='k', zorder=-1, alpha=0.9)
        vmin, vmax = cylim
        plt.subplot(gs[row, 1 + (col * 2)])
        cvals = get_cvals(epoch, d)
        k = get_pycortex_plot(cvals, 'fsaverage', vmin=vmin, vmax=vmax,
                              cmap=cortex_cmap)
        ax = plt.imshow(k)
        plt.xticks([])
        plt.yticks([])
        sns.despine(ax=plt.gca(), left=True, bottom=True, right=True, top=True)
    plt.savefig(
        '/Users/nwilming/Dropbox/UKE/confidence_study/all_signals_%s.svg' % suffix)
    plt.savefig(
        '/Users/nwilming/Dropbox/UKE/confidence_study/all_signals_%s.pdf' % suffix)


def table_performance(df, t_stim=1.3, t_resp=0, areas=None, sortby='MIDC_split',
                      signals=['MIDC_split', 'MIDC_nosplit', 'CONF_signed', 'CONF_unsigned', 'CONF_unsign_split']):
    '''
    Output a table of decoding performances, sorted by one signal.
    '''
    df_response = df.query("epoch=='response' & (latency==%f)" % t_resp).groupby(
        'signal').mean().T.loc[:, signals]
    df_response = df_response.sort_values(by=sortby, ascending=False)
    df_response.columns = pd.MultiIndex.from_tuples(
        [('response', x) for x in df_response.columns.values], names=['Epoch', 'Cluster'])
    df_stim = df.query("epoch=='stimulus' & (latency==%f)" % t_stim).groupby(
        'signal').mean().T.loc[df_response.index, signals]
    df_stim.columns = pd.MultiIndex.from_tuples(
        [('stimulus', x) for x in df_stim.columns.values], names=['Epoch', 'Cluster'])
    return pd.concat([df_response, df_stim], 1).round(2)


def make_state_space_triplet(df):
    plt.subplot(2, 2, 1)
    state_space_plot(df.test_roc_auc.Lateralized,   'MIDC_split',
                     'CONF_unsigned', df_b=df.test_roc_auc.Lateralized)
    plt.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.9, lw=1)
    plt.xlim([0.4, .9])
    plt.ylim([0.45, .7])
    plt.axhline(0.5, color='k', lw=1)
    plt.axvline(0.5, color='k', lw=1)
    plt.xlabel('Lateralized MIDC split')
    plt.ylabel('Lateralized CONF_unsigned')

    plt.subplot(2, 2, 2)
    state_space_plot(df.test_roc_auc.Lateralized,   'MIDC_split',
                     'CONF_unsigned', df_b=df.test_roc_auc.Averaged)
    plt.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.9, lw=1)
    plt.xlim([0.4, .9])
    plt.ylim([0.45, .7])
    plt.axhline(0.5, color='k', lw=1)
    plt.axvline(0.5, color='k', lw=1)
    plt.xlabel('Lateralized MIDC split')
    plt.ylabel('Averaged CONF_unsigned')
    plt.subplot(2, 2, 4)
    state_space_plot(df.test_roc_auc.Averaged,   'MIDC_split',
                     'CONF_unsigned', df_b=df.test_roc_auc.Averaged)
    plt.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.9, lw=1)
    plt.xlim([0.4, .9])
    plt.ylim([0.45, .7])
    plt.axhline(0.5, color='k', lw=1)
    plt.axvline(0.5, color='k', lw=1)
    plt.xlabel('Averaged MIDC split')
    plt.ylabel('Averaged CONF_unsigned')
    sns.despine(offset=5)
    plt.savefig(
        '/Users/nwilming/Dropbox/UKE/confidence_study/state_space_plot.svg')
    plt.savefig(
        '/Users/nwilming/Dropbox/UKE/confidence_study/state_space_plot.pdf')


def state_space_plot(df_a, signal_a, signal_b, df_b=None):
    def reshuffle(x, y):
        """Reshape the line represented by "x" and "y" into an array of individual
        segments.
        See: https://stackoverflow.com/questions/13622909/matplotlib-how-to-colorize-a-large-number-of-line-segments-as-independent-gradi/13649811#13649811
        """
        points = np.vstack([x, y]).T.reshape(-1, 1, 2)
        points = np.concatenate([points[:-1], points[1:]], axis=1)
        return points

    def plot_with_color(x, y, color, ax):
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap
        palette = LinearSegmentedColormap.from_list(
            'hrmpfh', [(1, 1, 1), color])
        segments = reshuffle(x, y)
        coll = LineCollection(segments, cmap=palette)
        coll.set_array(np.linspace(0, 1, len(x)))

        ax.add_collection(coll)

    palette = get_area_palette()
    # plt.clf()
    ax = plt.gca()  # ().add_subplot(111)
    #ax.plot([0.5, .5], [0, 1], 'k')
    #ax.plot([0, 1], [0.5, .5], 'k')
    if df_b is None:
        df_b = df_a
    for area in df_a.columns.get_level_values('cluster'):
        Aarea = pd.pivot_table(data=df_a.groupby(['signal', 'latency']).mean(),
                               index='signal', columns='latency', values=area)
        Barea = pd.pivot_table(data=df_b.groupby(['signal', 'latency']).mean(),
                               index='signal', columns='latency', values=area)
        plot_with_color(Aarea.loc[signal_a, :], Barea.loc[
            signal_b, :], palette[area], ax)

    # plt.show()


def get_area_palette(restrict=True):
    allc, vfc, glasser, jwg = ag.get_clusters()

    vfc_colors = sns.color_palette('Reds', n_colors=len(vfc) + 2)[1:-1]

    palette = {name: color for name, color in zip(
        list(vfc.keys()), vfc_colors) if name in ag.areas.values()}

    front_colors = sns.color_palette('Blues', n_colors=len(glasser) + 2)[1:-1]

    frontal = {name: color for name, color in zip(
        list(glasser.keys())[::-1],
        front_colors) if name in ag.areas.values()}

    jwdg_colors = sns.color_palette('Greens', n_colors=len(jwg) + 2)[1:-1]

    jwdg = {name: color for name, color in zip(
        list(jwg.keys())[::-1],
        jwdg_colors) if name in ag.areas.values()}
    palette.update(frontal)
    palette.update(jwdg)
    return palette


@memory.cache
def get_pycortex_plot(cvals, subject, vmin=0, vmax=1, cmap='RdBu_r'):
    import pymeg.source_reconstruction as pymegsr
    import pymeg.atlas_glasser as ag
    import cortex
    from scipy import misc
    labels = pymegsr.get_labels(subject=subject, filters=[
        '*wang*.label', '*JWDG*.label'], annotations=['HCPMMP1'])
    labels = pymegsr.labels_exclude(labels=labels, exclude_filters=[
        'wang2015atlas.IPS4', 'wang2015atlas.IPS5', 'wang2015atlas.SPL',
        'JWDG_lat_Unknown'])
    labels = pymegsr.labels_remove_overlap(
        labels=labels, priority_filters=['wang', 'JWDG'])

    V = ag.rois2vertex('fsaverage', cvals, 'lh', labels, vmin=vmin, vmax=vmax,
                       cmap=cmap)
    cortex.quickflat.make_png('/Users/nwilming/Desktop/test.png', V,
                              cutout='left', with_colorbar=False,
                              with_labels=False, with_rois=False)
    return misc.imread('/Users/nwilming/Desktop/test.png')


@memory.cache
def do_stats(x):
    from mne.stats import permutation_cluster_1samp_test
    return permutation_cluster_1samp_test(
        x, threshold=dict(start=0, step=0.2))


def ssd_overview_plot(ssd, area=['vfcPrimary', 'JWG_M1'], ylim=[0, 0.1]):
    import matplotlib
    signals = ['SSD', 'SSD_acc_contrast']
    gs = matplotlib.gridspec.GridSpec(len(area), 2)
    for j, a in enumerate(area):
        for i, signal in enumerate(signals):
            ax = plt.subplot(gs[j, i])
            plot_ssd_per_sample(ssd.query('signal=="%s"' %
                                          signal),
                                area=a,
                                ax=ax,
                                ylim=ylim)
            plt.title(signal)


def ssd_encoding_plot(ssd, ylim=[-0.01, 0.11]):
    import matplotlib
    signals = ['SSD', 'SSD_acc_contrast']
    gs = matplotlib.gridspec.GridSpec(2, 2)
    for i, signal in enumerate(signals):
        ax = plt.subplot(gs[0, i])
        plot_ssd_per_sample(ssd.Averaged.query('signal=="%s"' %
                                               signal),
                            area='vfcPrimary',
                            ax=ax,
                            ylim=ylim)
        plt.title(signal)
    for i, signal in enumerate(signals):
        ax = plt.subplot(gs[1, i])
        plot_ssd_per_sample(ssd.Lateralized.query('signal=="%s"' %
                                                  signal),
                            area='JWG_M1',
                            ax=ax,
                            ylim=ylim)


def plot_ssd_per_sample(ssd, area='vfcvisual', cmap='magma', alpha=0.05,
                        latency=0.18, save=False, ax=None, stats=True,
                        ylim=[0, 0.1]):
    '''
    '''
    import pylab as plt
    import seaborn as sns
    sns.set_style('ticks')
    cmap = plt.get_cmap(cmap)
    if ax is None:
        plt.figure(figsize=(6, 3.3))
        ax = plt.gca()

    for sample, ds in ssd.groupby('sample'):
        ds = ds.astype(float)
        if ds.columns.is_categorical():
            # Convert index to non-categorical to avoid pandas
            # bug? #19136
            ds.columns = pd.Index([x for x in ds.columns.values])
        k = ds.groupby(['subject', 'latency']).mean().reset_index()
        k = pd.pivot_table(ds.reset_index(), columns='latency',
                           index='subject', values=area).dropna()
        baseline = k.loc[:, :0].mean(axis=1)
        baseline_corrected = k.sub(baseline, axis=0)

        ax.plot(k.columns.values + 0.1 * sample,
                k.values.mean(0), color=cmap(sample / 10.))
        ax.set_ylim(ylim)
        if stats:
            t_obs, clusters, cluster_pv, H0 = do_stats(
                baseline_corrected.values)
            sig_x = (k.columns.values + 0.1 * sample)[cluster_pv < alpha]
            ax.plot(sig_x, sig_x * 0 - 0.0001 *
                    np.mod(sample, 2), color=cmap(sample / 10))
        ax.axvline(0.1 * sample + latency,
                   color='k', alpha=0.25, zorder=-1, lw=1)
    ax.set_xlabel(r'$time$')
    ax.set_ylabel(r'$contrast \sim power$')
    sns.despine(trim=True, ax=ax)
    if save:
        plt.tight_layout()
        plt.savefig(
            '/Users/nwilming/Dropbox/UKE/confidence_study/ssd_slopes_corr.svg')


def plot_ssd_peaks(peaks, palette, K=None):
    import pylab as plt
    import seaborn as sns
    from adjustText import adjust_text
    # from scipy.stats import
    plt.figure(figsize=(5.5, 4.5))
    pvt = peaks  # pd.pivot_table(peaks.astype(
    # float), index='sample', columns='subject')
    texts = []
    lines = []
    _, visual_field_clusters, _, _ = ag.get_clusters()
    for key, color in palette.items():
        try:
            apvt = pvt.loc[:, key]
        except KeyError:
            continue
        x, y = apvt.index.values, apvt.mean(1).values
        # Simple t-tests
        p = 1.0
        if K is not None:
            from scipy.stats import ttest_1samp
            Y = apvt.values
            corrs = []
            for sub in range(1, 16):
                corrs.append(
                    np.corrcoef(K.loc[:, sub].values,
                                apvt.loc[:, sub].values)[0, 1])
            _, p = ttest_1samp(np.tanh(corrs), 0)

        ps = plt.plot(x, y, color=color)

        if (key in visual_field_clusters) or ('cingulate_pos' in key):
            key = key.replace('vfc', '')
            key = key.replace('HCPMMP1_', '')
            if p < 0.05:
                key += '*'
            t = plt.text(x[-1], y[-1], key,
                         color=color, size=10)

        texts.append(t)
        lines.append(ps[0])

    plt.xlim(-1, 11)
    adjust_text(texts, only_move={'text': 'x', 'objects': 'x'},
                add_objects=lines,
                ha='center', va='bottom')
    plt.xlabel(r'$sample$')
    plt.ylabel(r'$slope$')
    sns.despine(trim=True)
    ax2 = plt.gca().twinx()
    ax2.plot(K.index.values, K.mean(1), 'k')
    ax2.set_ylim([-.1, 0.22])
    ax2.set_yticks([])
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(
        '/Users/nwilming/Dropbox/UKE/confidence_study/slopes_at_peakT_SSD.svg')


def extract_latency_peak_slope(ssd):
    ssd = ssd.query('epoch=="stimulus" & latency>=0 & Classifier=="Ridge"')
    ps = []
    index = []
    for idx, d in ssd.astype(float).groupby(['subject', 'sample']):
        levels = list(set(ssd.index.names) - set(['latency']))

        d.index = d.index.droplevel(levels)
        d = d.idxmax()
        ps.append(d)
        index.append(idx)
    ps = pd.concat(ps, 1).T
    return ps.set_index(pd.MultiIndex.from_tuples(index, names=['subject', 'sample']))


def extract_peak_slope(ssd, latency=0.18, dt=0.01, peak_latencies=None):
    if peak_latencies is None:
        ssd = ssd.query(
            'epoch=="stimulus" & %f<=latency & %f<=latency & Classifier=="Ridge"' % (
                latency - dt, latency + dt))

        ssd = ssd.astype(float).groupby(['subject', 'sample']).max()
        return pd.pivot_table(ssd, index='sample', columns='subject')
    else:
        reslist = []
        assert(len(ssd.index.get_level_values('signal').unique()) == 1)
        for idx, ds in ssd.groupby(['subject', 'sample']):            
            peak_idx = peak_latencies.loc[idx, :]            
            levels = list(set(ds.index.names) - set(['latency']))
            ds.index = ds.index.droplevel(levels)            
            res = {'subject': idx[0], 'sample': idx[1]}
            for col in ds.columns:
                latency = peak_idx.loc[col]
                value = ds.loc[latency, col]
                res.update({col: value})
            reslist.append(res)
        peaks = pd.DataFrame(reslist)
        peaks.set_index(['subject', 'sample'], inplace=True)
        print('.')
        return peaks#pd.pivot_table(ssd, index='sample', columns='subject')


def extract_kernels(dz, contrast_mean=0.0):
    """Kernels for each subject"""
    from conf_analysis.behavior import empirical

    K = (dz.groupby(['snum'])
         .apply(lambda x: empirical.get_pk(x, contrast_mean=contrast_mean,
                                           response_field='response'))
         .groupby(level=['snum', 'time'])
         .apply(lambda x: (x.query('optidx==1').mean()
                           - x.query('optidx==0').mean())))
    K.index.names = ['subject', 'sample']
    ks = pd.pivot_table(K, values='contrast',
                        index='sample', columns='subject')
    return ks


def plot_cluster_overview(decoding, ssd, tfr_resp, peaks, kernel, nf=True):
    '''
    Input should be a series.

    2 x 3 plot with
                SSD     TFR     Decoding
    stimulus
    response
    '''
    if nf:
        plt.figure(figsize=(12, 5))
    from conf_analysis.meg import srtfr
    from scipy.stats import linregress
    area = ssd.name
    assert(decoding.name == area)
    gs = gridspec.GridSpec(2, 3)
    tslice = {'stimulus': slice(-0.35, 1.35), 'response': slice(-1, 0.5)}
    xticks = {'stimulus': [-.35, 0, 0.5, 1, 1.35],
              'response': [-1, -.5, 0, 0.5]}
    cbars = []
    for i, epoch in enumerate(['stimulus', 'response']):
        if epoch == 'stimulus':
            plt.subplot(gs[i, 0])
            # SSD x stimulus
            plot_ssd_per_sample(ssd, area=area, cmap='magma', ax=plt.gca())
            ps = peaks.loc[:, area]
            K = kernel.mean(1).values
            P = ps.mean(1).values
            plt.plot(0.183 + (ps.index.values / 10.),
                     P, 'k', alpha=0.5)
            slope, inter, _, _, _ = linregress(K, P)
            plt.plot(0.183 + (kernel.index.values / 10.), slope * K + inter)

        plt.subplot(gs[i, 1])
        id_epoch = tfr_resp.index.get_level_values('epoch') == epoch

        s = srtfr.get_tfr_stack(
            tfr_resp.loc[id_epoch], area, tslice=tslice[epoch])
        t, p, H0 = srtfr.stats_test(s)
        p = p.reshape(t.shape)
        cbar = srtfr.plot_tfr(tfr_resp.loc[id_epoch], area, ps=p, minmax=5,
                              title_color=None, tslice=tslice[epoch])
        if epoch == 'stimulus':
            plt.axvline(-0.25, ls='--', color='k', alpha=0.5)
            plt.axvline(0., ls='--', color='k', alpha=0.5)
        if epoch == 'response':
            plt.xlabel(r'$time$')
        cbars.append(cbar)
        plt.xticks(xticks[epoch])
        plt.subplot(gs[i, 2])
        signals = pd.pivot_table(
            decoding.reset_index()
                    .query('epoch=="%s"' % epoch),
            columns='signal', index='latency',
            values=area).loc[tslice[epoch]]

        for col in signals:
            plt.plot(signals[col].index.values, signals[col].values, label=col)
        plt.ylabel(r'$AUC$')
        plt.xticks(xticks[epoch])
        plt.ylim([0.25, 0.75])
        if epoch == 'response':
            plt.xlabel(r'$time$')
    plt.legend()
    sns.despine(trim=True)
    for cbar in cbars:
        cbar.ax.yaxis.set_ticks_position('right')
    return cbar


def compare_kernel(K, peaks):
    res = {}
    for area in peaks.columns.get_level_values('cluster').unique():
        y = peaks[area]
        res[area] = [np.corrcoef(y.loc[:, i], K.loc[:, i])[
            0, 1] for i in K.columns]
    return res


def get_kernel_fits(kernel, peaks):
    from scipy.optimize import curve_fit
    from scipy.stats import linregress

    def func(x, a, c, d):
        return a * np.exp(-c * x) + d
    pars = []
    for sub in peaks.columns:
        x = np.arange(10)
        yk = kernel.loc[:, sub]  # - kernel.loc[:, sub].mean()
        yp = peaks.loc[:, sub]  # - peaks.loc[:, sub].mean()
        slope, inter, _, _, _ = linregress(yk, yp)
        yk = slope * yk + inter
        popt_k, pcov = curve_fit(func, x, yk, p0=(1, 1e-6, 1), maxfev=4000)
        popt_p, pcov = curve_fit(func, x, yp, p0=(1, 1e-6, 1), maxfev=4000)
        plt.subplot(3, 5, sub)
        plt.plot(x, yk, 'r')
        plt.plot(x, func(x, *popt_k), 'r--', alpha=0.5)
        plt.subplot(3, 5, sub)
        plt.plot(x, yp, 'b')
        plt.plot(x, func(x, *popt_p), 'b--', alpha=0.5)
        results = {'subject': sub, 'Ka': popt_k[0], 'Kc': popt_k[1],
                   'Kd': popt_k[2], 'Pa': popt_p[0], 'Pc': popt_p[1],
                   'Pd': popt_p[2]}
        pars.append(results)
    return pd.DataFrame(pars)


def make_r_data(area, peaks, k):
    ps = peaks.stack()
    ks = k.stack()
    ks.name = 'kernel'
    psks = ps.join(ks)
    P = psks.loc[:, ('kernel', area)].reset_index()
    return P


def fit_correlation_model(data, area):
    '''
    Use R to fit a bayesian hierarchical model that estimates correlation
    between time constants of kernels.
    '''
    from rpy2.robjects import r, pandas2ri
    pandas2ri.activate()
    from rpy2.robjects.packages import importr
    importr("brms")
    code = """
        prior1 <- (prior(normal(.2, 1), nlpar = "b1", resp = "kernel")
                  + prior(normal(0, 2), nlpar = "b2", resp="kernel")
                  + prior(normal(.2, 1), nlpar = "b1", resp = "{area}")
                  + prior(normal(0, 2), nlpar = "b2", resp="{area}"))

        FF <- bf(cbind(kernel, {area}) ~ b1 * exp(b2 * sample),
                b1 ~ 1|subject,
                b2~1|2|subject,
                nl = TRUE)
        fit1 <- brm(FF, data = P, prior = prior1,
                    control=list(adapt_delta=0.99),
                    cores=1, chains=2, iter=1000)
        random <- ranef(fit1)
        fixed <- fixef(fit1)
        summary(fit1)
    """.format(area=area)
    print(code)
    r.assign('P', data)
    df = r(code)
    return df, r


def plot_brain_color_legend(palette):
    '''
    Plot all ROIs on pysurfer brain. Colors given by palette.
    '''
    from surfer import Brain
    from pymeg import atlas_glasser as ag

    labels = sr.get_labels(subject='S04', filters=[
        '*wang*.label', '*JWDG*.label'], annotations=['HCPMMP1'])
    labels = sr.labels_exclude(labels=labels, exclude_filters=[
        'wang2015atlas.IPS4', 'wang2015atlas.IPS5', 'wang2015atlas.SPL',
        'JWDG_lat_Unknown'])
    labels = sr.labels_remove_overlap(
        labels=labels, priority_filters=['wang', 'JWDG'])
    lc = ag.labels2clusters(labels)
    brain = Brain('S04', 'lh', 'inflated',  views=['lat'], background='w')
    for cluster, labelobjects in lc.items():
        if cluster in palette.keys():
            color = palette[cluster]
            for l0 in labelobjects:
                if l0.hemi == 'lh':
                    brain.add_label(l0, color=color, alpha=1)
    # brain.save_montage('/Users/nwilming/Dropbox/UKE/confidence_study/brain_colorbar.png',
    #                   [[180., 90., 90.], [0., 90., -90.]])
    return brain
