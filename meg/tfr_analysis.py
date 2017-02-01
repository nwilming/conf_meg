import sys
sys.path.append('/home/nwilming/')
import glob
from conf_analysis.behavior import metadata
import mne, locale
import pandas as pd
import numpy as np
import cPickle
import json
import pylab as plt
import seaborn as sns
from pymeg import tfr
sns.set_style('ticks')
from joblib import Memory
memory = Memory(cachedir=metadata.cachedir)

def plot_contrast(df):
    '''
    Make a plot that shows the dependence of gamma and contrast for a specific contrast
    sample over time. In detail: plot contrast vs gamma.
    '''
    pass


def contrast_dependence(df, contrast, edges=np.linspace(0.1, 0.9, 15)):
    '''
    Compute contrast vs. power in a freq band.

    df is a data frame with time in columns and indexed by trial, channel and freq.
    Data in df is assumed to be baseline corrected.
    '''
    # Bring data frame into trial x time form.
    df = df.groupby(level='trial').mean()
    #df = df.join(contrast)
    centers = [low+(high-low)/2. for low, high in zip(edges[:-1], edges[1:])]
    #print df.columns
    return df.groupby(pd.cut(contrast, bins=edges, labels=centers)).mean()

def get_all_tfr():
    pass


def get_subject(snum, freq, channel, tmin, tmax,
                epoch='stimulus'):
    filenames = glob.glob('/home/nwilming/conf_meg/S%i/*%stfr.hdf5'%(snum, epoch))
    metafiles = glob.glob('/home/nwilming/conf_meg/S%i/*%s.meta'%(snum, epoch))
    avg = tfr.get_tfrs(filenames, freq=freq, channel=channel, tmin=tmin, tmax=tmax)
    avg.loc[:, 'snum'] = snum
    avg.set_index('snum', append=True, inplace=True)
    meta = pd.concat([pd.read_hdf(f) for f in metafiles])
    return avg, meta


def baseline(avg, id_base):
    '''
    Baseline correction by dividing by average baseline
    '''
    base = avg.loc[:, id_base].values.ravel().mean()
    avg = np.log10(avg / base)
    return avg


def avg_baseline(avg, baseline):
    '''
    Baseline correction by dividing by average baseline
    '''
    time =  avg.columns.get_level_values('time').values.astype(float)
    id_base = (baseline[0]<time) & (time<baseline[1])
    #print id_base
    crs = []
    avg = avg.apply(np.log10)
    # Baseline correction with average baseline across all trials. (base is freq specific)
    base = avg.loc[:, id_base].groupby(level='freq').mean().mean(1) # This should be len(#Freq)
    def div(x):
        bval = base.loc[x.index.get_level_values('freq').values[0]]
        return x - bval
    return avg.groupby(level='freq').apply(div)


def trial_baseline(avg, baseline):
    '''
    Baseline correction by dividing by per-trial baseline
    '''
    index = avg.index
    time =  avg.columns.get_level_values('time').values.astype(float)
    id_base = (baseline[0]<time) & (time<baseline[1])
    avg = np.log10(avg)
    base = avg.loc[:, id_base].groupby(level=['freq', 'trial']).mean().mean(1)
    levels = list(set(avg.index.names) - set(baseline.index.names))
    avg.index = avg.index.droplevel(levels)
    avg.div(base, axis=0, inplace=True)
    avg.index = index
    return avg


def avg_contrast_resp(avg, meta, edges=np.linspace(0.2, 0.8, 7),
                      cidx=slice(0, 10),
                      baseline=(-0.4, 0), per_group_baseline=False,
                      baseline_func=avg_baseline
                      ):
    '''
    avg needs to be indexed by freq and trial hash
    '''
    n_groups = len(edges)-1
    cv = np.vstack(meta.contrast_probe)[:, cidx].mean(1)
    contrast = pd.Series(cv, index=
                         meta.index.get_level_values('hash'))
    grouper = (pd.cut(contrast, edges, labels=np.arange(n_groups))
                        .reset_index())
    crs = []

    if not per_group_baseline:
        avg = baseline_func(avg, baseline)

    for i, (c, o) in enumerate(grouper.groupby(0)):
        m = (avg.loc[(slice(None), list(o.hash.values)), :]
             .groupby(level='freq')
             .mean())
        if per_group_baseline:
            m = baseline_func(m, baseline)
        m.loc[:, 'cgroup'] = c
        m.set_index('cgroup', append=True, inplace=True)
        crs.append(m)
    return pd.concat(crs)


def plot_by_freq(resp, **kwargs):
    if 'vmin' not in kwargs.keys():
        kwargs['vmin'] = -0.5
    if 'vmax' not in kwargs.keys():
        kwargs['vmax'] = 0.5
    if 'cmap' not in kwargs.keys():
        kwargs['cmap'] = 'RdBu_r'

    n_groups = len(np.unique(resp.index.get_level_values('cgroup')))
    for i, (c, m) in enumerate(resp.groupby(level='cgroup')):
        time = np.unique(m.columns.get_level_values('time'))
        tc = np.array([time[0]] + list([(low+(high-low)/2.)
            for low, high in zip(time[:-1], time[1:])]) + [time[-1]])
        freqs = np.unique(m.index.get_level_values('freq'))
        fc = np.array([freqs[0]] + list([(low+(high-low)/2.)
            for low, high in zip(freqs[:-1], freqs[1:])]) + [freqs[-1]])

        plt.subplot(1, n_groups, i+1)
        print m.values.min().min(), m.values.max().max()
        plt.pcolormesh(tc, fc,
                m.values, **kwargs)
        #plt.xlim([-0.4, 1.1])
        plt.ylim([min(freqs), max(freqs)])
        if i>0:
            plt.yticks([])
        else:
            plt.ylabel('Frequency')
            plt.xlabel('Time')
        plt.xticks([-.4, 0, .4, .8])


def plot_avg_freq(resp, freqs=slice(60, 90)):
    n_groups = len(np.unique(resp.index.get_level_values('cgroup')))
    colors = sns.color_palette("coolwarm", n_groups)
    if np.mod(n_groups, 2):
        del colors[len(colors)/2]
    for i, (c, m) in enumerate(resp.groupby(level='cgroup')):
        time = m.columns.get_level_values('time')
        m = m.loc[(freqs,), :]
        y = m.values.mean(0)
        plt.plot(time, y, label=c, color=colors[i])
        #plt.xlim([-0.4, 1.1])
        plt.xlabel('Time')
        plt.xticks([-.4, 0, .4, .8])
