'''
Analyze behavior of single trials
'''
import numpy as np
import os
import pandas as pd

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing
from glob import glob
from os.path import join
from pymeg import aggregate_sr as asr

from joblib import Memory


if 'TMPDIR' in os.environ.keys():
    memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'])
    inpath = '/nfs/nwilming/MEG/sr_labeled/aggs'
    outpath = '/nfs/nwilming/MEG/sr_decoding/'
elif 'RRZ_LOCAL_TMPDIR' in os.environ.keys():
    tmpdir = os.environ['RRZ_LOCAL_TMPDIR']
    outpath = '/work/faty014/MEG/sr_labeled/aggs/'
    outpath = '/work/faty014/MEG/sr_decoding/'
    memory = Memory(cachedir=tmpdir)
else:
    inpath = '/home/nwilming/conf_meg/sr_labeled/aggs'
    outpath = '/home/nwilming/conf_meg/sr_decoding'
    memory = Memory(cachedir=metadata.cachedir)


def get_trials(subject, epoch, cluster, hemi):
    filenames = glob(join(inpath, 'S%i_*_%s_agg.hdf' % (subject, epoch)))
    meta = preprocessing.get_meta_for_subject(subject, 'stimulus')
    agg = asr.delayed_agg(filenames, hemi=hemi, cluster=cluster)()
    return meta, agg


def samplewise(agg, pl):
    res = []
    for (sub, cluster), ds in agg.groupby(['subject', 'cluster']):
        latencies = pl.loc[sub, cluster] + np.arange(0, 1, 0.1)
        res.append(ds.loc[:, latencies])
    return pd.concat(res)


def featmat(agg):
    '''
    convert to trial X (time*freq)
    '''
    return pd.pivot_table(index='trial', columns='freq', data=agg)


def trials_by_column(data, col):
    '''
    Takes data, df that contains all trials for one cluster,
    and sorts by series col.
    '''
    # Make sure data has only trials as index
    names = data.index.names
    for name in names:
        if name == 'trial':
            continue
        data.index = data.index.droplevel(name)

    svals = col.index
    data = data.loc[svals, :]
    assert(all(data.index.values == svals.values))
    return data


def trials_by_index(data, col):
    # Make sure data has only trials as index
    names = data.index.names
    for name in names:
        if name == 'trial':
            continue
        data.index = data.index.droplevel(name)

    svals = col.index
    data = data.loc[svals, :]
    assert(all(data.index.values == svals.values))
    return data


def auc(data, meta, field):
    res = {}
    for idx, df in data.groupby(['freq']):
        res[idx] = kernel(
            df.values,
            meta.loc[df.index.get_level_values('trial'), field])
    return pd.DataFrame(res, index=data.columns.get_level_values('time')).T


def kernel(data, choices):
    '''
    Compute ROC AUC for choice weights.
    '''
    from sklearn.metrics import roc_auc_score
    kernel = [roc_auc_score(choices, column) for column in data.T]
    return kernel


def get_all_cia(pl, freqbins=[0, 10, 35, 70, 150], hemi='Averaged', cluster='vfcPrimary'):
    freqbins = np.asarray(freqbins)
    centers = (freqbins[1:] - freqbins[:-1]) / 2
    frames = []
    for subject in range(1, 16):
        print(subject)
        meta, agg = get_trials(subject, 'stimulus', cluster, hemi)
        cutter = pd.cut(agg.index.get_level_values(
            'freq'), freqbins, labels=centers)
        agg = agg.groupby([cutter, 'trial']).mean()
        agg.index.names = ['bfreq', 'trial']
        for sample in range(10):
            peak = pl[(subject, sample)].values[0, 0]
            t = (agg.loc[:, peak]
                    .groupby('bfreq')
                    .apply(lambda x: contrast_integrated_averages(
                        x, meta, sample, centers=np.linspace(0.2, 0.8, 21)))
                    .unstack())
            t.loc[:, 'subject'] = subject
            t.loc[:, 'sample'] = sample
            t.loc[:, 'cluster'] = cluster
            t.loc[:, 'hemi'] = hemi
            t.set_index(['subject', 'hemi', 'cluster',
                         'sample', 'bfreq'], inplace=True)
            frames.append(t)
    return pd.concat(frames)


def contrast_integrated_averages(agg, meta, sample, centers=np.linspace(0.1, 0.9, 5),
                                 width=0.2):
    '''
    For each colum in agg compute contrast integrated average
    '''
    agg = agg.groupby('trial').mean()
    contrast = np.stack(meta.loc[agg.index.values, 'contrast_probe'])[
        :, sample]
    w = width / 2.
    rows = []
    for center in centers:
        idx = ((center - w) < contrast) & (contrast < (center + w))
        r = agg.loc[idx, :].mean()
        # r.loc['contrast'] = center
        # r.set_index('contrast', inplace=True, append=True)
        rows.append(r)
    rows = pd.concat(rows, 1).T
    rows = rows.set_index(centers)
    rows.index.name = 'contrast'
    c50 = ((0.5 - w) < contrast) & (0.5 < (center + w))
    r = agg.loc[c50, :].mean().values
    return pd.DataFrame(rows.values - r, index=rows.index,
                        columns=rows.columns)


def cia(power, contrast, centers=np.linspace(0.1, 0.9, 5), width=0.2):
    w = width / 2.0
    rows = []
    for center in centers:
        idx = ((center - w) < contrast) & (contrast < (center + w))
        r = power[idx].mean()
        rows.append(r)
    return centers, np.array(rows)

    
def plot_crfs(crfs, freqbins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    import seaborn as sns
    import pylab as plt
    freqbins = np.asarray(freqbins)
    freq = crfs.index.get_level_values('freq')
    centers = freqbins[0:-1] + np.diff(freqbins)
    cutter = pd.cut(freq, bins=freqbins, labels=centers).astype(int)
    crfs = crfs.groupby(
        [cutter, 'sample', 'subject', 'contrast', 'hemi', 'cluster']).mean()
    crfs.index.names = ['freq', 'sample', 'subject', 'contrast', 'hemi', 'cluster']
    g = sns.relplot(data=crfs.reset_index(), row='hemi',
                    col='cluster', hue='sample', kind='line',                    
                    x='contrast', y='power', ci=None)
    #g.map(sns.lineplot, 'contrast', 'power', units='subject')

    return g


def std_corr(data, crfs):
    from conf_analysis.behavior import empirical
    from scipy.stats import linregress
    bias = (data.groupby(['snum'])
            .apply(empirical.crit).reset_index())
    bias.columns = ['subject', 'bias']
    bias.set_index('subject', inplace=True)
    dp = (data.groupby(['snum'])
          .apply(empirical.dp).reset_index())
    dp.columns = ['subject', 'dp']
    dp.set_index('subject', inplace=True)

    #crfs = crfs.groupby(['subject', 'freq', 'contrast']).mean()

    crfsd = crfs.groupby(['subject', 'freq']).max() - \
        crfs.groupby(['subject', 'freq']).min()
    crfsd.name = 'power'
    crfsd = pd.pivot_table(crfsd.reset_index(),
                           columns='freq',
                           index='subject',
                           values='power')

    #crfs = crfs.groupby(['subject', 'bfreq']).mean()
    contrast = crfs.index.get_level_values('contrast')

    crfs = (crfs
            .loc[(0.6 < contrast) & (contrast < 0.7)]
            .groupby(['subject', 'freq'])
            .mean()) - \
        (np.abs(
            crfs
            .loc[(0.3 < contrast) & (contrast < 0.4)]
            .groupby(['subject', 'freq'])
            .mean()))
    crfs.name = 'power'
    crfs = pd.pivot_table(crfs.reset_index(),
                          columns='freq',
                          index='subject',
                          values='power')

    results = []
    for freq in crfs:
        df = crfs.loc[:, freq]
        dff = crfsd.loc[:, freq]
        slope, _, d, p_d, _ = linregress(
            dp.loc[df.index, 'dp'].values, dff.values)
        slope, _, c, p_c, _ = linregress(bias.loc[df.index, 'bias'], df)
        results.append({'freq': freq, 'dp': d, 'bias': c,
                        'pval_dp': p_d, 'pval_bias': p_c})
    return pd.DataFrame(results)
