import numpy as np
import pylab as plt
import pandas as pd
from conf_analysis.meg import preprocessing
import seaborn as sns
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter


def get_power(subject):
    df = pd.read_hdf(
        '/home/nwilming/conf_data/S%02i-stimulus-srcpow.hdf' % subject)
    df = df.set_index(['time', 'trial'])
    del df['snum']
    meta = preprocessing.get_meta_for_subject(subject, 'stimulus')
    return df, meta


def get_sample_traces(df, area, meta, sample, edges):
    '''
    Return power traces grouped by contrast (defined by edges)
    '''
    ko = df.loc[:, area]
    ko = ko.unstack('time').loc[:, 0.5:]
    ko.columns = ko.columns - 0.75
    ko = ko - ko.mean()
    cvals = pd.Series(np.vstack(meta.loc[ko.index.values, 'contrast_probe'])[
                      :, sample], index=ko.index)
    cvals.name = 'S3c'
    ko.index = cvals
    return ko.groupby(pd.cut(ko.index, edges)).mean()


def plot_sample_traces(df, meta):
    from scipy.ndimage import gaussian_filter

    edges = [0] + list(np.linspace(0.3, 0.7, 5)) + [1]
    plt.figure(figsize=(12, 6))
    for sample in range(10):
        plt.subplot(2, 5, sample + 1)
        s = get_sample_traces(df, 'max-LO1-lh', meta, sample, edges)
        for row in s.values:
            plt.plot(s.columns.values, gaussian_filter(row, 15))
        plt.legend([])
    sns.despine()
    plt.tight_layout()


def get_correlations(df, meta, area):
    '''
    Compute correlations between power and contrast per time point and
    contrast sample
    '''
    stuff = []
    darea = pd.pivot_table(df.copy(), values=area,
                           index='trial', columns='time')
    darea = darea.subtract(darea.loc[:, 0.5:0.75].mean(1), 'index')
    cvals = np.vstack(meta.loc[darea.index.values, 'contrast_probe'])
    for cp in range(10):
        res = {}
        cc = [np.corrcoef(cvals[:, cp], darea.loc[:, col].values)[0, 1]
              for col in darea]
        res['area'] = area
        res['sample'] = cp
        res['corr'] = cc
        res['time'] = darea.columns.values
        stuff.append(pd.DataFrame(res))
    return pd.concat(stuff)


def get_all_correlations(df, meta, n_jobs=12):
    generator = (delayed(get_correlations)(
        df.loc[:, area].reset_index(),
        meta,
        area)
        for area in df.columns)
    cc = Parallel(n_jobs=n_jobs)(generator)
    return pd.concat(cc)


def plot_area(cc, **kwargs):
    vals = []
    for d, c in cc.groupby(['sample']):
        time = c.time.values - 0.75
        idx = (0.1 * d <= time) & (time < (0.1 * d + 0.4))
        vals.append(c.loc[idx, 'corr'])
        plt.plot(time[idx] - (0.1 * d), #(0.1 * d) +
                 c.loc[idx, 'corr'], **kwargs)
        plt.xlim([-0.25, 0.5])
        #yl = [(0.1 * d) - 0.1, (0.1 * d) + 0.1]
        plt.plot([0, 0], (-0.2, 0.2), color='k', alpha=0.5)
    vals = np.stack(vals).mean(0)
    plt.plot(time[idx] - (0.1*d), vals, lw=3.5, color='k', alpha=0.75)


def sample_aligned_power(df, meta, area):
    '''
    Create a dataframe that has contrast as index and time in the colums.
    time is aligned to sample onset.
    '''
    darea = pd.pivot_table(df, values=area,
                           index='trial', columns='time')
    darea = darea.subtract(darea.loc[:, 0.5:0.75].mean(1), 'index')
    darea = darea.subtract(darea.mean(0))
    cvals = np.vstack(meta.loc[darea.index.values, 'contrast_probe'])
    stuff = []
    for i, trial in enumerate(darea.index.values):
        for ii, sample in enumerate(cvals[i, :]):
            res = {}
            power = darea.loc[trial, sample*0.1:sample*0.1+0.4]
            res['power'] = power.values
            res['time'] = np.around(
                power.index.values - power.index.values.min(), 5)
            res['contrast'] = sample
            res['trial'] = trial
            res['sample'] = ii
            stuff.append(pd.DataFrame(res))
    return pd.concat(stuff)
