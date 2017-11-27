import numpy as np
import pylab as plt
import pandas as pd
from conf_analysis.meg import preprocessing
from conf_analysis.behavior import metadata
import seaborn as sns
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
from joblib import Memory
from glob import glob
from sklearn import cluster, neighbors
from sklearn.metrics import pairwise
from conf_analysis.meg import tfr_analysis as ta

memory = Memory(cachedir=metadata.cachedir)


def get_power(subject, F, decim=3):
    files = glob('/home/nwilming/conf_meg/S%i-SESS*-F%i*-lcmv.hdf' %
                 (subject, F))
    df = pd.concat([pd.read_hdf(f) for f in files])

    if decim is not None:
        def decim_func(df, x):
            df = df.reset_index().set_index('time')
            df = df.sort_index()
            return df.loc[slice(0, len(df), x)]
        df = df.groupby('trial').apply(lambda x: decim_func(x, decim))
        del df['trial']
    df = df.reset_index().set_index(['time', 'trial'])
    df = combine_areas(df, hemi=True)

    meta = preprocessing.get_meta_for_subject(subject, 'stimulus')
    return df, meta


def get_sample_traces(df, area, meta, sample, edges, baseline=(-0.25, 0)):
    '''
    Return power traces grouped by contrast (defined by edges)
    '''
    ko = df.loc[:, area]
    ko = ko.unstack('time').loc[:, -0.25:1.4]
    ko.columns = ko.columns
    #ko = ko - ko.mean()
    cvals = pd.Series(np.vstack(meta.loc[ko.index.values, 'contrast_probe'])[
                      :, sample], index=ko.index)
    cvals.name = 'S3c'
    ko.index = cvals
    values = ko.groupby(pd.cut(ko.index, edges)).mean()
    base = values.loc[:, slice(*baseline)].mean(1)
    values = values.subtract(base, axis=0)
    return values


def plot_sample_traces(df, meta, area):
    from scipy.ndimage import gaussian_filter

    edges = [0] + list(np.linspace(0.3, 0.7, 3)) + [1]
    plt.figure(figsize=(12, 6))
    for sample in range(10):
        plt.subplot(2, 5, sample + 1)
        s = get_sample_traces(df, area, meta, sample, edges)
        for i, row in enumerate(s.values):
            plt.plot(s.columns.values, row, label=s.index.values[i])

        plt.axvline(sample * 0.1, color='k')
    plt.legend()
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
        plt.plot(time[idx] - (0.1 * d),  # (0.1 * d) +
                 c.loc[idx, 'corr'], **kwargs)
        plt.xlim([-0.25, 0.5])
        #yl = [(0.1 * d) - 0.1, (0.1 * d) + 0.1]
        plt.plot([0, 0], (-0.2, 0.2), color='k', alpha=0.5)
    vals = np.stack(vals).mean(0)
    plt.plot(time[idx] - (0.1 * d), vals, lw=3.5, color='k', alpha=0.75)


@memory.cache
def sample_aligned_power(df, meta, area):
    '''
    Create a dataframe that has contrast as index and time in the colums.
    time is aligned to sample onset.
    '''
    darea = pd.pivot_table(df.reset_index(), values=area,
                           index='trial', columns='time')
    #darea = np.log10(darea)
    darea = darea.subtract(darea.loc[:, -0.25:0].mean(1), 'index')
    darea = darea.subtract(darea.mean(0))
    cvals = np.vstack(meta.loc[darea.index.values, 'contrast_probe'])
    stuff = []
    for i, trial in enumerate(darea.index.values):
        for ii, sample in enumerate(cvals[i, :]):
            res = {}
            power = darea.loc[trial, (sample * 0.1) - 0.1:sample * 0.1 + 0.4]
            base = power.loc[(sample * 0.1) - 0.1:0].mean()
            res['power'] = power.values - base
            res['time'] = np.around(
                power.index.values - power.index.values.min(), 5) - 0.1
            res['contrast'] = sample
            res['trial'] = trial
            res['sample'] = ii
            stuff.append(pd.DataFrame(res))
    return pd.concat(stuff)


def plot_sample_aligned_power(stuff, edges=[0, 0.5, 1], ax=None):
    k = stuff.groupby(['sample', 'time', pd.cut(stuff.contrast, edges)]).mean()
    del k['contrast']
    pd.pivot_table(k, index='contrast', columns='time',
                   values='power').T.plot(ax=ax)


def plot_sample_aligned_power_all_areas(df, meta, edges):
    import matplotlib
    gs = matplotlib.gridspec.GridSpec(5, 4)
    areas = df.columns
    for i, area in enumerate(areas):
        row, col = np.mod(i, 5), i // 5
        plt.subplot(gs[row, col])
        s = sample_aligned_power(df, meta, area)
        plot_sample_aligned_power(s, edges, ax=plt.gca())
        plt.title(area)
        plt.axvline(0, color='k', alpha=0.5)
        # plt.set_yscale('log')
        if row == 4:
            plt.xticks([0, 0.2, 0.4])
            plt.xlabel('')
        else:
            plt.xticks([0, 0.2, 0.4], [])
            plt.xlabel('time')
        if col == 0:
            plt.ylabel('Power')
            #plt.yticks([-5, 0, 5])
        else:
            plt.ylabel('')
            #plt.yticks([-5, 0, 5], [])
        #plt.ylim([-10, 10])
        plt.legend([])
        plt.text(0, 7.5, area)
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    import seaborn as sns
    sns.despine()
    return gs


def combine_areas(df, hemi=False):
    areas = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d',
             'hV4', 'VO2', 'PHC1', 'PHC2',
             'TO1', 'LO1', 'LO2', 'V3A', 'IPS0', 'IPS1', 'IPS3',
             'IPS4', 'IPS5', 'FEF']
    res = []

    def foo(df, areas, hemi=''):
        res = []
        for area in areas:
            cols = [c for c in df.columns if (area in c)]
            name = area + hemi
            if len(cols) >= 1:
                col = df.loc[:, cols].mean(1)
                col.name = name
                res.append(col)
        return res
    if hemi:
        res.extend(
            foo(df.loc[:, [x for x in df.columns if 'rh' in x]], areas, hemi='rh'))
        res.extend(
            foo(df.loc[:, [x for x in df.columns if 'lh' in x]], areas, hemi='lh'))
    else:
        res.extend(foo(df, areas))

    return pd.concat(res, axis=1)


def get_dfp_for_all_subs():
    frames = []
    for subject in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]:
        df, meta = get_power(subject, 59)
        dfp = combine_areas(df)
        areas = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d',
                 'hV4', 'VO2', 'PHC1', 'PHC2',
                 'TO1', 'LO1', 'LO2', 'V3A', 'IPS0', 'IPS1', 'IPS3',
                 'IPS4', 'IPS5', 'FEF']
        for i, area in enumerate(areas):
            saligned = sample_aligned_power(dfp, meta, area)
            saligned.loc[:, 'subject'] = subject
            saligned.loc[:, 'area'] = area
            frames.append(saligned)
    return frames


def make_fir_row(times,
                 event_times=np.arange(0, 1, 0.1),
                 event_length=0.4,
                 Hz=600):
    ets = np.linspace(0, event_length, event_length * Hz)
    # Make a single row for a FIR deconvolution
    # Design matrix is #times x #event_length*Hz
    dm = np.zeros((len(times), int(event_length * Hz)))
    for i, t in enumerate(times):
        for st in event_times:
            if (st < t) or (t < st + event_length):
                idx = np.argmin(abs(t - st - ets))
                dm[i, idx] = 1
    return dm


def make_fir_dm(times):
    template = make_fir_row(np.unique(times))
    dm = np.zeros((len(times), template.shape[1]))
    for i, t in enumerate(times):
        idx = np.argmin(abs(times - t))
        dm[i, :] = template[idx, :]
    return dm
