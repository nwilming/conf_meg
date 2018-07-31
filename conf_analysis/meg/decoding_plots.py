import numpy as np
import os
import pandas as pd
import pylab as plt
import seaborn as sns

from matplotlib import cm
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import pyplot
from pymeg import roi_clusters as rois
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


def filter_latency(data, min, max):
    lat = data.index.get_level_values('latency').values
    return data.loc[(min < lat) & (lat < max), :]


def make_brain_plots(data, atype='Pairs', ssd_view=['cau']):
    # 1 AUC
    auc_limits = {'MIDC_split': (0.3, 0.7), 'MIDC_nosplit': (0.3, 0.7),
                  'CONF_signed': (0.3, 0.7), 'CONF_unsigned': (.3, 0.7),
                  'SIDE_nosplit': (0.3, 0.7)}
    '''
    df = data.test_roc_auc.Pairs.query(
        'epoch=="stimulus" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, 0.15, 0.2),
                         limits=auc_limits, epoch='stimulus', measure='auc')
    df = data.test_roc_auc.Pairs.query(
        'epoch=="response" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, -0.05, 0.05),
                         limits=auc_limits, epoch='response', measure='auc')

    df = data.test_roc_auc.loc[:, atype].query(
        'epoch=="response" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, 0.425, 0.475),
                         limits=auc_limits, epoch='response_late_' + 'atype', measure='auc')

    acc_limits = {'MIDC_split': (-0.2, 0.2), 'MIDC_nosplit': (-0.2, 0.2),
                  'CONF_signed': (-0.2, 0.2), 'CONF_unsigned': (-.2, 0.2),
                  'SIDE_nosplit': (-0.2, 0.2)}
    df = data.test_accuracy.Pairs.query(
        'epoch=="stimulus" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, -0.05, 0.05),
                         limits=acc_limits, epoch='stimulus', measure='accuracy')
    df = data.test_accuracy.Pairs.query(
        'epoch=="response" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, -0.05, 0.05),
                         limits=acc_limits, epoch='response', measure='accuracy')
    '''
    data_ssd = filter_latency(data.test_slope.Pairs.query(
        'epoch=="stimulus" & (signal=="SSD") & Classifier=="Ridge"'), 0.18, 0.19)
    for sample, sd in data_ssd.groupby('sample'):
        ssd_limits = {'SSD': (-0.08, 0.08)}
        plot_summary_results(sd, limits=ssd_limits,
                             epoch='stimulus',
                             measure='slope' + '_sample%i' % sample,
                             views=ssd_view)


def plot_summary_results(data, cmap='RdBu_r',
                         limits={'MIDC_split': (0.3, 0.7),
                                 'MIDC_nosplit': (0.3, 0.7),
                                 'CONF_signed': (0.3, 0.7),
                                 'CONF_unsigned': (.3, 0.7),
                                 'SIDE_nosplit': (0.3, 0.7),
                                 'SSD': (-0.05, 0.05)},
                         ex_sub='S04', measure='auc', epoch='response',
                         classifier='svc',
                         views=[['par', 'fro'], ['lat', 'med']]):
    from pymeg import roi_clusters as rois, source_reconstruction as sr

    # labels = sr.get_labels(ex_sub)
    labels = sr.get_labels(ex_sub)
    lc = rois.labels_to_clusters(labels, rois.all_clusters, hemi='lh')

    for signal, dsignal in data.groupby('signal'):
        vmin, vmax = limits[signal]
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        colortable = cm.get_cmap(cmap)
        cfunc = lambda x: colortable(norm(x))
        brain = plot_one_brain(dsignal, signal, lc, cfunc, ex_sub=ex_sub,
                               measure=measure, classifier=classifier,
                               epoch=epoch, views=views)
    return brain


#@memory.cache
def plot_one_brain(dsignal, signal, lc, cmap, ex_sub='S04', classifier='SCVlin',
                   epoch='response', measure='auc', views=[['par', 'fro'], ['lat', 'med']]):
    from surfer import Brain
    print('Creating Brain')
    brain = Brain(ex_sub, 'lh', 'inflated',  views=['lat'], background='w')
    # subjects_dir='/Users/nwilming/u/freesurfer_subjects/')
    print('Created Brain')
    ms = dsignal.mean()
    if (signal == 'CONF_signed') and (measure == 'accuracy'):
        plot_labels_on_brain(brain, lc, ms, cmap)
    if (signal == 'SSD'):
        plot_labels_on_brain(brain, lc, ms, cmap)
    else:
        plot_labels_on_brain(brain, lc, ms, cmap)
    brain.save_montage('/Users/nwilming/Desktop/%s_montage_%s_%s_%s.png' %
                       (signal, measure, classifier, epoch), views)
    return brain


def plot_labels_on_brain(brain, labels, data, cmap):
    already_plotted = []
    for label in data.index.values:
        for clustername, lobjects in labels.items():
            if clustername == label:
                for l0 in lobjects:
                    if any(l0.name == x for x in already_plotted):
                        import pdb
                        pdb.set_trace()
                    # print(('Addding', l0.name, cmap(value)))
                    already_plotted.append(l0.name)
                    value = data.loc[label]
                    l0.color = cmap(value)
                    brain.add_label(l0, color=cmap(value), alpha=0.8)

    brain.save_image('test.png')


def plot_brain_color_legend(data, palette):
    from surfer import Brain
    labels = sr.get_labels('S04')
    lc = rois.labels_to_clusters(labels, rois.all_clusters, hemi='lh')
    brain = Brain('S04', 'lh', 'inflated',  views=['lat'], background='w')
    for cluster, labelobjects in lc.items():
        color = palette[cluster]
        for l0 in labelobjects:
            brain.add_label(l0, color=color, alpha=1)
    brain.save_montage('/Users/nwilming/Dropbox/UKE/confidence_study/brain_colorbar.png',
                       [['par', 'fro'], ['lat', 'med']])
    return brain


@memory.cache
def _dcd_helper_getter(path):
    df = pd.read_hdf(os.path.join(
        path, 'all_decoded_samples_with_split_conf.hdf'))
    df = df.query('~(signal=="SSD")')
    df.loc[:, 'latency'] = np.around(df.latency.astype(float), 4)
    df = df.reset_index().set_index(
        ['Classifier', 'epoch', 'est_key', 'latency', 'mc<0.5', 'signal', 'subject', 'area'])

    df = df.loc[~df.index.duplicated(), :]
    df = df.unstack('area').T
    return df


@memory.cache
def _ssd_helper_getter(path):
    ssd = pd.read_hdf(os.path.join(
        path, 'all_decoded_samples_with_split_conf.hdf'))
    ssd = ssd.query('(signal=="SSD")')
    ssd.loc[:, 'latency'] = np.around(ssd.latency.astype(float), 4)
    ssd = ssd.reset_index().set_index(
        ['Classifier', 'epoch', 'est_key', 'latency', 'signal', 'subject', 'sample', 'area'])
    ssd = ssd.loc[~ssd.index.duplicated(), :]
    ssd = ssd.unstack('area').T
    return ssd


@memory.cache
def _ssd_add_helper_getter(path):
    ssd = pd.read_hdf(os.path.join(
        path, 'SSD_additional.hdf'))
    ssd.loc[:, 'latency'] = np.around(ssd.latency.astype(float), 4)
    ssd = ssd.reset_index().set_index(
        ['Classifier', 'epoch', 'est_key', 'latency', 'signal', 'subject', 'sample', 'area'])
    ssd = ssd.loc[~ssd.index.duplicated(), :]
    ssd = ssd.unstack('area').T
    return ssd


def recode(df):
    dt = []
    areas = []
    for area in df.index.get_level_values('area'):
        if '_L-R' in area:
            dt.append('Lateralized')
            areas.append(area.replace('-lh_L-R', ''))
        elif '_Havg' in area:
            dt.append('Average')
            areas.append(area.replace('-lh_Havg', ''))
        else:
            dt.append('Pairs')
            areas.append(area.split(',')[0].replace(
                '(', '').replace(')', '').replace("'", '').replace('-lh', ''))
    df.loc[:, 'atype'] = dt
    df.set_index('atype', append=True, inplace=True)
    df = df.swaplevel(1, 2)
    df = df.T
    df.columns.set_levels(areas, level='area', inplace=True)
    return df


def get_decoding_data(path='/home/nwilming/conf_meg/', ssd_classifier="Ridge",
                      decoding_classifier="SCVlin"):
    # files = glob.glob(os.path.join(path, 'concat_S*'))
    # df = pd.concat([pd.read_hdf(f) for f in files])
    df = recode(_dcd_helper_getter(path))
    ssd = recode(_ssd_helper_getter(path))
    ssdadd = recode(_ssd_add_helper_getter(path))
    # ssd = df.query('signal=="SSD"')
    # df = df.query('~(signal=="SSD")')
    idt = df.test_accuracy.index.get_level_values('signal') == 'CONF_signed'
    df.loc[idt, 'test_accuracy'] = (df.loc[idt, 'test_accuracy'] - 0.25).values
    df.loc[~idt, 'test_accuracy'] = (
        df.loc[~idt, 'test_accuracy'] - 0.5).values
    return (df.query('~(subject==6) & Classifier=="%s"' % decoding_classifier),
            ssd.query("Classifier=='%s'" % ssd_classifier),
            ssdadd.query("Classifier=='%s'" % ssd_classifier))


def plot_signals(data, measure, classifier='svm', ylim=(0.45, 0.75)):
    for epoch, de in data.groupby('epoch'):
        print(de.shape)
        g = plot_by_signal(de)
        g.set_ylabels(r'$%s$' % measure)
        g.set_xlabels(r'$time$')
        g.set(ylim=ylim)
        plt.savefig('/Users/nwilming/Desktop/%s_%s_%s_decoding.pdf' %
                    (measure, classifier, epoch))


def plot_signals_hand(data, palette, measure, classifier='svm',  midc_ylim=(-0.25, 0.25),
                      conf_ylim=(-0.05, 0.25), **kw):
    '''
    Make signal plot by hand to get layout right

    '''
    col_order = list(rois.glasser.keys()) + \
        list(rois.visual_field_clusters.keys()) + list(rois.jwrois.keys())
    plt.figure(figsize=(9, 10))
    combinations = [
        ('stimulus', 'MIDC_nosplit'), ('stimulus',
                                       'MIDC_split'), ('response', 'MIDC_split'),
        (None, None), ('stimulus', 'CONF_signed'), ('response', 'CONF_signed'),
        (None, None), ('stimulus', 'CONF_unsigned'), ('response', 'CONF_unsigned')]
    index = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
             (1, 2), (2, 0), (2, 1), (2, 2)]
    gs = gridspec.GridSpec(3, 3)

    for i, (epoch, signal) in enumerate(combinations):
        if epoch is None:
            continue
        row, col = index[i]
        plt.subplot(gs[row, col])
        d = data.query('epoch=="%s" & signal=="%s"' %
                       (epoch, signal))
        if '_split' in signal:
            d = d.groupby(['latency', 'mc<0.5']).mean()
            for split, ds in d.groupby('mc<0.5'):
                for column in col_order:
                    try:
                        latency = ds.index.get_level_values('latency')
                        if not split:
                            plt.plot(latency,
                                     -ds.loc[:, column].values, color=palette[column],
                                     **kw)
                        else:
                            plt.plot(latency,
                                     ds.loc[:, column].values, color=palette[
                                         column],
                                     **kw)

                    except KeyError:
                        pass

        else:

            d = d.groupby(['latency']).mean()
            for column in col_order:
                try:
                    plt.plot(d.index.values, d.loc[
                             :, column], color=palette[column], **kw)
                except KeyError:
                    pass

        if 'MIDC' in signal:
            plt.ylim(midc_ylim)
            pyplot.locator_params(nticks=5)
        else:
            plt.ylim(conf_ylim)
            pyplot.locator_params(nticks=5)

        if (col == 0) or ((col == 1) and (row > 0)):
            plt.ylabel(r'$%s$' % measure)
        if (row == 2) or ((row == 0) and (col == 0)):
            plt.xlabel(r'$time$')
        center = (plt.xlim()[1] + plt.xlim()[0]) / 2.
        plt.text(plt.xlim()[0] + 0.05, 0.18, signal, size=8)

        sns.despine(trim=True)
        if epoch == 'response':
            ts = plt.yticks()[0]
            print(ts)
            plt.fill_between([-0.05, 0.05], ts[0], ts[-1], color='gray',
                             zorder=-100, alpha=0.75)
    plt.savefig('/Users/nwilming/Dropbox/UKE/confidence_study/all_signals.svg')


def get_area_palette(areas):
    nvfc = len(rois.visual_field_clusters.keys())
    vfc_colors = sns.color_palette('Reds', n_colors=nvfc + 2)[1:-1]
    # sns.cubehelix_palette(
    # nvfc, start=1.8, rot=0, dark=.2, light=.7, hue=1, gamma=1.1)

    palette = {name: color for name, color in zip(
        list(rois.visual_field_clusters.keys())[::-1], vfc_colors)}

    nfront = len(rois.glasser.keys())
    front_colors = sns.color_palette('Blues', n_colors=nfront + 2)[1:-1]
    # sns.cubehelix_palette(
    # nfront, start=2.6, rot=0, dark=.4, light=.7, hue=1, gamma=1.1)

    frontal = {name: color for name, color in zip(
        list(rois.glasser.keys())[::-1],
        front_colors)}

    njwdg = len(rois.jwrois.keys())

    jwdg_colors = sns.color_palette('Greens', n_colors=njwdg + 2)[1:-1]
    # sns.cubehelix_palette(
    # njwdg, start=0.6, rot=0, dark=.4, light=.9, hue=1, gamma=1.1)
    jwdg = {name: color for name, color in zip(
        list(rois.jwrois.keys())[::-1],
        jwdg_colors)}
    palette.update(frontal)
    palette.update(jwdg)
    return palette


def plot_by_signal(data, signals={'MIDC_split': '#E9003A', 'MIDC_nosplit': '#FF5300',
                                  'CONF_signed': '#00AB6F', 'CONF_unsign_split': '#58E000',
                                  'CONF_unsigned': '#58E000'}):
    palette = get_area_palette(data.columns)

    idsig = data.index.get_level_values('signal').isin(signals.keys())
    data = data.loc[idsig, :]
    split = data.index.get_level_values('mc<0.5').values.astype(float)

    nosplit = data.loc[np.isnan(split), :].groupby(
        ['latency', 'signal']).mean().stack().reset_index()

    concat_df = [nosplit]

    for split_ind, dsignal in data.loc[~np.isnan(split)].groupby('mc<0.5'):
        k = dsignal.groupby(['latency', 'signal']
                            ).mean().stack().reset_index()
        if not split_ind:

            k.loc[:, 0] *= -1
        k.columns = ['latency', 'signal', 'area', split_ind]
        concat_df.append(k)
    k = pd.concat(concat_df)

    g = sns.FacetGrid(k, col='signal', col_wrap=2, hue='area', palette=palette)
    g.map(plt.plot, 'latency', 0, alpha=0.7, lw=1)
    g.map(plt.plot, 'latency', 1, alpha=0.7, lw=1)

    #    k = data.groupby(['latency', 'signal']).mean().stack().reset_index()
    #    g = sns.FacetGrid(k, col='signal', col_wrap=2,
    #                      hue='area', palette='magma')
    #    g.map(plt.plot, 'latency', 0, alpha=0.8)
    return g


def plot_all_signals(df):
    plot_signals(df.test_roc_auc.Average.query(
        'Classifier=="SCVlin"') - 0.5, measure='\Delta AUC',
        classifier='svmlin_avgs', ylim=(-0.25, 0.25))

    plot_signals(df.test_roc_auc.Pairs.query(
        'Classifier=="SCVlin"') - 0.5, measure='\Delta AUC',
        classifier='svmlin_pairs', ylim=(-0.25, 0.25))

    plot_signals(df.test_accuracy.Average.query(
        'Classifier=="SCVlin"'), measure='\Delta Accuracy',
        classifier='svmlin_avgs', ylim=(-0.25, 0.25))

    plot_signals(df.test_accuracy.Pairs.query(
        'Classifier=="SCVlin"'), measure='\Delta Accuracy',
        classifier='svmlin_pairs', ylim=(-0.25, 0.25))


def plot_interesting_areas(data,
                           signals={'MIDC_split': '#E9003A', 'SIDE_nosplit': '#FF5300',
                                    'CONF_signed': '#00AB6F', 'CONF_unsigned': '#58E000'},
                           title='', classifier='SCVlin'):
    interesting_areas = ['visual', 'FEF', 'IPS_Pces', 'M1', 'aIPS1', 'Area6_dorsal_medial',
                         'Area6_anterior', 'A6si', 'PEF', '55b', '8av', '8C', '24dv']
    areas = []

    areas = [x for x in data.columns if any(
        [i in x for i in interesting_areas])]

    data = data.query('Classifier=="%s"' % classifier)
    plot_set(areas, data, title=title)


def plot_set(area_set, data,
             signals={'MIDC_split': '#E9003A', 'SIDE_nosplit': '#FF5300',
                      'CONF_signed': '#00AB6F', 'CONF_unsigned': '#58E000'},
             title=''):

    gs = gridspec.GridSpec(1, 2)
    for i, area in enumerate(area_set):
        for signal, color in signals.items():
            try:
                plot_decoding_results(data, signal, area, stim_ax=gs[
                    0, 0], resp_ax=gs[0, 1], color=color,
                    offset=i * 0.5)
            except RuntimeError:
                print('RuntimeError for area %s, signal %s' % (area, signal))
    sns.despine(left=True)
    plt.suptitle(title)

    # Add legend
    x = [-0.5, -1, -1, -0.5]
    y = [0., 0.2, 0, 0.2]
    ylim = list(plt.ylim())
    ylim[1] = len(area_set) * 0.5 + 1
    for i, (signal, color) in enumerate(signals.items()):
        plt.text(x[i], y[i], signal, color=color)
    plt.subplot(gs[0, 0])
    plt.ylim(ylim[0] - 0.1, ylim[1])
    plt.subplot(gs[0, 1])
    plt.ylim(ylim[0] - 0.1, ylim[1])


def plot_decoding_results(data, signal, area,
                          stim_ax=None, resp_ax=None,  color='b',
                          offset=0):
    '''
    Data is a df that has areas as columns and at least subjct, classifier, latency and signal as index.
    Values of the dataframe encode the measure of choice to plot.
    '''
    import warnings
    warnings.filterwarnings("ignore")
    if stim_ax is None:
        stim_ax = plt.gca()
    if resp_ax is None:
        stim_ax = plt.gca()
    data = data.loc[:, area]
    select_string = 'signal=="%s"' % (signal)
    areaname = (str(area).replace('vfc', '')
                .replace('-lh', '')
                .replace('-rh', '')
                .replace('_Havg', '')
                .replace('_Lateralized', ''))
    data = data.reset_index().query(select_string)
    if '_split' in signal:
        data = data.groupby(['subject', 'epoch', 'latency']
                            ).mean().reset_index()
    stimulus = data.query('epoch=="stimulus"').reset_index()
    stimulus.loc[:, area] += offset

    response = data.query('epoch=="response"').reset_index()
    response.loc[:, area] += offset
    stim_ax = plt.subplot(stim_ax)
    sns.tsplot(stimulus, time='latency', value=area,
               unit='subject', ax=stim_ax, color=color)
    # plt.ylim([0.1, 0.9])
    plt.axhline(0.5 + offset, color='k')
    dx, dy = np.array([0.0, 0.0]), np.array([.5, 0.75])
    plt.plot(dx, dy + offset, color='k')
    plt.text(-0.75, 0.6 + offset, areaname)
    plt.yticks([])
    plt.ylabel('')
    resp_ax = plt.subplot(resp_ax)

    sns.tsplot(response, time='latency', value=area,
               unit='subject', ax=resp_ax, color=color)
    plt.plot(dx, dy + offset, color='k')
    plt.yticks([])
    plt.ylabel('')
    # plt.ylim([0.1, 0.9])
    plt.axhline(0.5 + offset, color='k')


@memory.cache
def do_stats(x):
    from mne.stats import permutation_cluster_1samp_test
    return permutation_cluster_1samp_test(
        x, threshold=dict(start=0, step=0.2))


def plot_ssd_per_sample(ssd, area='vfcvisual', cmap='magma', alpha=0.05,
                        latency=0.18, save=False, ax=None, stats=True):
    import pylab as plt
    import seaborn as sns
    sns.set_style('ticks')
    cmap = plt.get_cmap(cmap)
    if ax is None:
        plt.figure(figsize=(6, 3.3))
        ax = plt.gca()

    for sample, ds in ssd.groupby('sample'):
        ds = ds.astype(float)
        k = ds.groupby(['subject', 'latency']).mean().reset_index()
        k = pd.pivot_table(ds.reset_index(), columns='latency',
                           index='subject', values=area).dropna()
        baseline = k.loc[:, :0].mean(axis=1)
        baseline_corrected = k.sub(baseline, axis=0)

        ax.plot(k.columns.values + 0.1 * sample,
                k.values.mean(0), color=cmap(sample / 10.))
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


def plot_ssd_peaks(peaks, palette):
    import pylab as plt
    import seaborn as sns
    from adjustText import adjust_text
    # from scipy.stats import
    plt.figure(figsize=(4.5, 3.5))
    pvt = pd.pivot_table(peaks.astype(
        float), index='sample', columns='subject')
    texts = []
    lines = []
    for key, color in palette.items():
        try:
            apvt = pvt.loc[:, key]
        except KeyError:
            continue
        x, y = apvt.index.values, apvt.mean(1).values

        ps = plt.plot(x, y, color=color)

        if key in rois.visual_field_clusters:
            t = plt.text(x[-1], y[-1], key.replace('vfc', ''),
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
    plt.tight_layout()
    plt.savefig(
        '/Users/nwilming/Dropbox/UKE/confidence_study/slopes_at_peakT_SSD.svg')


def extract_latency_slope(ssd, latency=0.18):
    ulat = np.unique(ssd.index.get_level_values('latency'))
    latency = ulat[np.argmin(np.abs(ulat - latency))]

    ssd = ssd.query('epoch=="stimulus" & latency==%f & Classifier=="Ridge"' %
                    latency).test_slope.Average

    ssd = ssd.astype(float).groupby(['subject', 'sample']).mean()
    return pd.pivot_table(ssd, index='sample', columns='subject')


def extract_peak_slope(ssd):
    ssd = ssd.query(
        'epoch=="stimulus" & latency>0 & Classifier=="Ridge"')

    ssd = ssd.astype(float).groupby(['subject', 'sample']).max()
    return pd.pivot_table(ssd, index='sample', columns='subject')


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
    from mne.stats import permutation_t_test
    from scipy.stats import linregress, ttest_1samp
    # K = K.values.mean(1)
    ccs = []
    kxs = []
    ps = []
    for sub in K.columns:
        kx = K.loc[:, sub]
        slope, inter, _, _, _ = linregress(kx, peaks.values.mean(1))
        kx = slope * kx + inter
        y = peaks.loc[:, sub]
        ps.append(y)
        kxs.append(kx)
        ccs.append(np.corrcoef(kx, y)[0, 1])
    #_, p, _ = permutation_t_test(np.array([ccs]).T)
    ccs = np.array(ccs)
    return ttest_1samp(ccs, 0)
    return np.stack(kxs).mean(0), np.stack(ps).mean(0), p


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
