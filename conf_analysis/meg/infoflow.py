'''
Analyze behavior of single trials
'''
import numpy as np
import os
import pandas as pd

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing
from glob import glob
from joblib import Memory
from os.path import join
from pymeg import aggregate_sr as asr


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
    meta = preprocessing.get_meta_for_subject(
        subject, 'stimulus').set_index('hash')
    agg = asr.delayed_agg(filenames, hemi=hemi, cluster=cluster)()
    idx = pd.Series([subject] * len(agg), index=agg.index)
    idx.name = 'subject'
    agg.set_index(idx, append=True, inplace=True)
    return meta, agg


def samplewise(agg, pl):
    res = []
    for (subject, cluster), d in agg.groupby(['subject', 'cluster']):
        latencies = pl.loc[(subject,), cluster].values + np.arange(0, 1, 0.1)
        res.append(d.loc[:, latencies])
    return pd.concat(res)


def featmat(agg):
    '''
    Convert to n_trials x (time + freq)
    '''
    mats = []
    trials = agg.index.get_level_values('trial').unique()
    freqs = []
    samples = []
    for f, df in agg.groupby('freq'):
        k = trials_by_index(df, trials)
        mats.append(k)
        freqs.append(k.values[0, :] * 0 + f)
        s = np.arange(10)
        samples.append(s)

    return trials, np.hstack(freqs), np.hstack(samples), np.hstack(mats)


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

    data = data.loc[col, :]
    assert(all(data.index.values == col.values))
    return data


def classifier_by_freq(agg, meta):
    from sklearn.linear_model import LogisticRegression
    coefs = []
    for f, df in agg.groupby('freq'):
        trials, freqs, samples, fm = featmat(df)
        clf_d = (LogisticRegression(max_iter=20000, class_weight='balanced')
                 .fit(fm, meta.loc[trials].response.values))
        coefs.append(clf_d.coef_)
    return coefs


def avg_freqs(data, bins=np.array([0, 10, 25, 75, 150])):
    bins = np.asarray(bins)
    ctrs = (bins[1:] + bins[:-1]) / 2

    cutter = pd.cut(data.index.get_level_values('freq'), bins,
                    labels=ctrs)
    cutter.name = 'freq'
    dp = data.groupby([cutter, 'subject', 'cluster', 'trial']).mean()
    dp.index.names = ['freq', 'subject', 'cluster', 'trial']
    return dp


@memory.cache()
def all_auc_by_TFR(cluster, hemi, epoch):
    res = []
    for subject in range(1, 16):
        meta, agg = get_trials(subject, epoch, cluster, hemi)
        r = get_auc_by_TFR(agg, meta.response)
        r.set_index(agg.columns.get_level_values('time'), inplace=True)
        r.loc[:, 'subject'] = subject
        res.append(r.set_index('subject', append=True))
    return pd.concat(res)


@memory.cache()
def get_auc_by_TFR(data, choices, bins=np.array([0, 10, 25, 75, 150])):
    ctrs = (bins[1:] + bins[:-1]) / 2
    cutter = pd.cut(data.index.get_level_values('freq'), bins,
                    labels=ctrs)
    res = {}
    data = data.groupby([cutter, 'trial']).mean()

    for center in ctrs:
        df = data.loc[(center, slice(None)), :]
        idx = df.index.get_level_values('trial')
        c = choices.loc[idx].values
        res[center] = kernel(df.values, c)
    return pd.DataFrame(res)


def kernel(data, choices):
    '''
    Compute ROC AUC for choice weights.
    '''
    from sklearn.metrics import roc_auc_score
    kernel = [roc_auc_score(choices, column) for column in data.T]
    return kernel
