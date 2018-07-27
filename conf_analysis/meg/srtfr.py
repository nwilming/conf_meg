

'''
Compute TFRs from source reconstructed data

This module works on average TFRs for specific ROIS. It takes the output from
source reconstruction and reduces this to average TFRs.

The logical path through this module is:

1) load_sub_grouped reduces RAW data to average tfrs. It's a good idea
   to call this function for each subject on the cluster to distribute
   memory load. Output will be cached.
2) To compute a contrast call load_sub_grouped with a filter_dict. This
   allows to compute average TFRs for sub groups of trials per subject.
3) Use the plotting functions to plot TFRs for different ROIs.
'''

import os
import pandas as pd
from conf_analysis.meg import preprocessing
from pymeg.contrast_tfr import Cache, compute_contrast, augment_data
from pymeg import parallel
from joblib import Memory


memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'], verbose=0)


contrasts = {
    'all': (['all'], [1]),
    'choice': (['hit', 'fa', 'miss', 'cr'], (1, 1, -1, -1)),
    'stimulus': (['hit', 'fa', 'miss', 'cr'], (1, -1, 1, -1)),
    'hand': (['left', 'right'], (0.5, -0.5)),
}


def submit_contrasts(collect=False):
    tasks = []
    for subject in range(1, 2):
        for session in range(0, 4):
            tasks.append((contrasts, 'lat', subject, session, 'response'))
            tasks.append((contrasts,  'avg', subject, session, 'response'))
            tasks.append((contrasts, 'lat', subject, session, 'stimulus'))
            tasks.append((contrasts, 'avg', subject, session, 'stimulus'))
    return [_eval(get_contrasts, task, collect=collect, walltime=2)
            for task in tasks]


def _eval(func, args, collect=False, **kw):
    """
    Intermediate helper to toggle cluster vs non cluster
    """
    if not collect:
        parallel.pmap(get_contrast, [args], **kw)
    else:
        return get_contrast(*args)


@memory.cache
def get_contrasts(contrasts, hemi, subject, session, epoch):
    contrast_store = []
    if hemi == 'lat':
        if subject < 8:
            hemi = 'lh_is_ipsi'
        else:
            hemi = 'rh_is_ipsi'
    with Cache() as cache:
        for (contrast, (conditions, weights)) in contrasts.items():

            res = get_contrast(contrast, conditions, weights,
                               hemi, subject, session, epoch, cache)
            contrast_store.append(res)
    return pd.concat(contrast_store)


@memory.cache(ignore=['cache'])
def get_contrast(name, conditions, weights, hemi, subject, session, epoch, cache):
    stim = '/nfs/nwilming/MEG/sr_labeled/S%i-SESS%i-stimulus*.hdf' % (
        subject, session)
    resp = '/nfs/nwilming/MEG/sr_labeled/S%i-SESS%i-response*.hdf' % (
        subject, session)

    meta = preprocessing.get_meta_for_subject(subject, 'stimulus')
    response_left = meta.response == 1
    stimulus = meta.side == 1
    meta = augment_data(meta, response_left, stimulus)
    if session == 'stimulus':
        contrast = compute_contrast(conditions, weights, hemi, stim, stim,
                                    meta, (-0.25, 0), n_jobs=15, cache=cache)
    else:
        contrast = compute_contrast(conditions, weights, hemi, resp, stim,
                                    meta, (-0.25, 0), n_jobs=15, cache=cache)
    contrast.loc[:, 'subject'] = subject
    contrast.loc[:, 'session'] = session
    contrast.loc[:, 'contrast'] = name
    contrast.loc[:, 'hemi'] = hemi
    contrast.set_index(['subject', 'session', 'contrast',
                        'hemi'], append=True, inplace=True)
    return contrast


# Ignore following for now

def plot_labels(data, areas, locations, gs, stats=True, minmax=(10, 20),
                tslice=slice(-0.25, 1.35)):
    '''
    Plot TFRS for a set of ROIs. At most 6 labels.
    '''
    labels = rois.filter_cols(data.columns, areas)
    import pylab as plt
    # import seaborn as sns
    # colors = sns.color_palette('bright', len(labels))

    p = None
    maxrow = max([row for row, col in locations])
    maxcol = max([row for row, col in locations])

    for (row, col), area in zip(locations, labels):

        plt.subplot(gs[row, col])
        ex_tfr = get_tfr(data.query('est_key=="F"'), area, tslice=tslice)
        s = get_tfr_stack(data.query('est_key=="F"'), area, tslice=tslice)
        if stats:
            t, p, H0 = stats_test(s)
            p = p.reshape(t.shape)
        cbar = _plot_tfr(area, ex_tfr.columns.values, ex_tfr.index.values,
                         s.mean(0), p, title_color='k', minmax=minmax[0])
        if ((row + 2, col + 1) == gs.get_geometry()):
            pass
        else:
            cbar.remove()
        plt.xticks([])

        if col > 0:
            plt.yticks([])
        else:
            plt.ylabel('Freq')
        plt.subplot(gs[row + 1, col])
        ex_tfr = get_tfr(data.query('est_key=="LF"'), area, tslice=tslice)
        s = get_tfr_stack(data.query('est_key=="LF"'), area, tslice=tslice)
        if stats:
            t, p, H0 = stats_test(s)
            p = p.reshape(t.shape)
        cbar = _plot_tfr(area, ex_tfr.columns.values, ex_tfr.index.values,
                         s.mean(0), p, title_color='k', minmax=minmax[1])
        cbar.remove()
        # plt.xticks([0, 0.5, 1])
        if row == maxrow:
            plt.xlabel('time')

            # plt.xticks([tslice.start, 0, tslice.stop])
        else:
            plt.xticks([])
