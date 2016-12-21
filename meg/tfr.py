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

def contrast_response(avg, meta, grouper, edges, toi=(0, 0.1), baseline=(-0.1, 0)):
    '''
    Average is a dataframe that is indexed by trial and has time points as
    columns.
    '''
    groups = pd.cut(grouper, edges, labels=edges[:-1] + (np.diff(edges)/2.))
    #groups = pd.DataFrame({'cgroups':groups.to_dense()}, index=avg.index)
    avg.loc[:, 'cgroup'] = groups.to_dense()
    avg = avg.set_index('cgroup', append=True)
    def cr(group, baseline, toi):
        time = group.columns.get_level_values('time').values
        idx = (baseline[0]<= time) & (time < baseline[1])
        base =  group.loc[:, idx].mean().mean() # Dim: num_trials
        idx = (toi[0]<= time) & (time < toi[1])
        data = group.loc[:, idx] # Dim: num_trials x time
        #data = data.div(base, axis=0)
        return data

    # Compute baseline corrected contrast for each groups
    return avg.groupby(level='cgroup', as_index=True, group_keys=True).apply(lambda x: cr(x, baseline, toi))
