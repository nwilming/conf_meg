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
sns.set_style('ticks')

def cr_plot(avg, meta, dt=0.1, db=0.1):
    cvals = np.vstack(meta.contrast_probe)-0.5
    edges = np.percentile(cvals.ravel(), np.arange(0, 101, 20))
    colors = sns.color_palette(n_colors=len(edges)-1)
    labels = edges[:-1] + (np.diff(edges)/2.)
    colors = dict( (l,c) for l,c in zip(labels, colors))

    for idx, toi in zip(range(10), np.arange(0, 1, 0.1)):
        plt.subplot(2, 5, idx+1)
        k = contrast_response(avg, meta, cvals[:, idx], edges, toi=(toi, toi+dt), baseline=(toi-db, toi))
        for i, (g, l) in enumerate(k.groupby(level='cgroup').mean().iterrows()):
            plt.plot(l.index.get_level_values('time'), l.values, label=np.around(g, 2), color=colors[g])
            plt.legend()
    sns.despine()
    plt.tight_layout()


def allcr(avg, meta, dt=0.1, db=0.1, bins=5, events=np.arange(0, 10)):
    cvals = np.vstack(meta.contrast_probe)-0.5
    edges = np.percentile(cvals.ravel(), np.linspace(0, 100, bins+1))
    colors = sns.color_palette(n_colors=len(edges)-1)
    labels = edges[:-1] + (np.diff(edges)/2.)
    crs = []
    event_t = np.arange(0, 1, 0.1)[events]
    print edges
    for idx, toi in zip(events, event_t):
        if db is not None:
            db = (toi-db, toi)
            if db[0] > db[1]:
                db = db[::-1]
        k = contrast_response(avg, meta, cvals[:, idx], edges, 
                              toi=(toi, toi+dt), baseline=db,
                              labels=labels)#np.arange(len(edges)-1))
        tlabels = k.columns.get_level_values('time')
        time_res = np.diff(tlabels)[0]
        new_labels = dict((k, np.around(i*time_res, 4)) for i, k in enumerate(tlabels))
        k = k.rename(columns=new_labels)
        crs.append(k)
    return pd.concat(crs)


def contrast_response(avg, meta, grouper, edges, toi=(0, 0.1), baseline=(-0.1, 0), labels=None):
    '''
    Average is a dataframe that is indexed by trial and has time points as
    columns.
    '''
    column_index = avg.columns
    if labels is None:
        labels = edges[:-1] + (np.diff(edges)/2.)
    groups = pd.cut(grouper, edges, labels=labels)
    #groups = pd.DataFrame({'cgroups':groups.to_dense()}, index=avg.index)
    avg.loc[:, 'cgroup'] = groups.to_dense()
    avg.set_index('cgroup', append=True, inplace=True)
    def cr(group, baseline, toi):
        time = group.columns.get_level_values('time').values
        idx = (toi[0]<= time) & (time < toi[1])
        data = group.loc[:, idx]# Dim: num_trials x time
        if baseline is not None:
            idx = (baseline[0]<= time) & (time < baseline[1])
            base =  group.loc[:, idx].mean().mean() # Dim: num_trials
            data = data-base
        return data

    # Compute baseline corrected contrast for each groups
    cr =  avg.groupby(level='cgroup', as_index=True, group_keys=True).apply(lambda x: cr(x, baseline, toi))
    avg.index = avg.index.droplevel(1)
    avg.columns = column_index 
    return cr


def avg_contrast_resp(avg, meta, edges=np.linspace(0.2, 0.8, 7), 
                      cidx=slice(0, 10),
                      baseline=(-0.4, 0)):
    '''
    avg needs to be indexed by freq and trial hash
    '''
    n_groups = len(edges)-1
    cv = np.vstack(meta.contrast_probe)[:, cidx].mean(1)
    contrast = pd.Series(cv, index=
                         meta.index.get_level_values('hash'))
    grouper = (pd.cut(contrast, edges, labels=np.arange(n_groups))
                        .reset_index())
    time =  avg.columns.get_level_values('time').values.astype(float)
    id_base = (baseline[0]<time) & (time<baseline[1])
    crs = []
    for i, (c, o) in enumerate(grouper.groupby(0)):
        m = (avg.loc[(slice(None), list(o.hash.values)), :]
             .groupby(level='freq')
             .mean())
        base = m.loc[:, id_base].mean(1)
        m = m.div(base, axis=0)-1
        m.loc[:, 'cgroup'] = c
        m.set_index('cgroup', append=True, inplace=True)
        crs.append(m)
    return pd.concat(crs)


def plot_by_freq(resp):
    n_groups = len(np.unique(resp.index.get_level_values('cgroup')))
    for i, (c, m) in enumerate(resp.groupby(level='cgroup')):
        time = m.columns.get_level_values('time')
        freqs = m.index.get_level_values('freq')
        plt.subplot(1, n_groups, i+1)       
        plt.pcolormesh(time, freqs,
                m.values, vmin=-0.5, vmax=0.5, cmap='RdBu_r')
        plt.xlim([-0.4, 1.1])
        plt.ylim([20, 140])
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
        print y.shape, time.shape
        plt.plot(time, y, label=c, color=colors[i])
        plt.xlim([-0.4, 1.1])
        plt.xlabel('Time')
        plt.xticks([-.4, 0, .4, .8])