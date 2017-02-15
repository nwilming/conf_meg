import sys
sys.path.append('/home/nwilming/')
import glob
import mne, locale
import pandas as pd
import numpy as np
import cPickle
import json
import pylab as plt
import seaborn as sns
from pymeg import tfr
sns.set_style('ticks')
import time


def get_subs(freq, channel, tmin, tmax, epoch='stimulus', subs=range(1, 16)):
    avgs = []
    metas = []
    for snum in subs:
        start = time.time()
        filenames = glob.glob('/home/nwilming/conf_meg/S%i/*%stfr.hdf5'%(snum, epoch))
        metafiles = glob.glob('/home/nwilming/conf_meg/S%i/*%s.meta'%(snum, epoch))
        df = tfr.get_tfrs(filenames, freq=freq, channel=channel, tmin=tmin, tmax=tmax)
        print 'S%i loading %2.1fs'%(snum, time.time()-start)
        avg = df.groupby(level=['freq', 'trial']).mean()
        avg.loc[:, 'snum'] = snum
        avg.set_index('snum', append=True, inplace=True)
        print 'S%i grouping and setting index %2.1fs'%(snum, time.time()-start)
        avgs.append(avg)
        metas.append(pd.concat([pd.read_hdf(f) for f in metafiles]))
        print 'S%i took %2.1fs'%(snum, time.time()-start)
    return avgs, metas


def avg_baseline(avg, baseline):
    '''
    Baseline correction by dividing by average baseline
    '''
    time =  avg.columns.get_level_values('time').values.astype(float)
    id_base = (baseline[0]<time) & (time<baseline[1])
    crs = []
    # Baseline correction with average baseline across all trials. (base is freq specific)
    base = avg.loc[:, id_base].groupby(level='freq').mean().mean(1) # This should be len(#Freq)
    div = lambda x: np.log10((x / base.loc[x.index.get_level_values('freq').values[0]]))
    return avg.groupby(level='freq').apply(div)


def trial_baseline(avg, baseline):
    '''
    Baseline correction by dividing by per-trial baseline
    '''
    time =  avg.columns.get_level_values('time').values.astype(float)
    id_base = (baseline[0]<time) & (time<baseline[1])
    crs = []
    # Baseline correction with average baseline across all trials. (base is freq specific)
    avg = np.log10(avg)
    base = avg.loc[:, id_base].groupby(level=['freq', 'trial']).mean().mean(1)
    return avg.sub(base, axis=0)


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
