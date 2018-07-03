

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
from glob import glob
if 'DISPLAY' in list(os.environ.keys()):
    try:
        from surfer import Brain
    except:
        Brain = None
        print('No pysurfer support')

import numpy as np
import pandas as pd

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing
from pymeg import roi_clusters as rois

from joblib import Memory

memory = Memory(cachedir=metadata.cachedir)

'''
The following functions will reduce raw data to various different average TFRs.
'''


def transcode_columns(data):
    data.columns.name = 'area'
    areas, dt = [], []
    for area in data.columns:
        if '_Lateralized' in area:
            dt.append('Lateralized')
        elif '_Havg' in area:
            dt.append('Average')
        else:
            dt.append('Single')
        areas.append(area.replace(
            '-lh_Havg', '').replace('-lh_Lateralized', ''))
    idx = pd.MultiIndex.from_tuples(zip(dt, areas),
                                    names=['atype', 'area'])
    data.columns = idx.sort_values()
    #data.columns = data.columns.sort_values()
    return data


def baseline(data, baseline_data, baseline=(-0.25, 0)):
    baseline_data = baseline_data.query('%f < time & time < %f' % baseline)
    m = baseline_data.mean()
    s = baseline_data.std()
    # print(m, s)
    return (data.subtract(m, 1)
                .div(s, 1))


def confidence_contrast(subs=list(range(1, 16)), epoch='stimulus'):
    return pd.concat(
        [_prewarm_confidence_contrast(sub, epoch=epoch) for sub in subs])


def _prewarm_confidence_contrast(sub, epoch='stimulus', baseline_time=(-0.25, 0)):
    meta = preprocessing.get_meta_for_subject(sub, 'stimulus')
    filter_dict = {'C0': meta.query('confidence==0').reset_index().loc[:, 'hash'].values,
                   'C1': meta.query('confidence==1').reset_index().loc[:, 'hash'].values}
    return contrast(sub, filter_dict, None, None,
                    epoch=epoch, baseline_time=baseline_time)


def contrast_controlled_response_contrast(sub, epoch='stimulus',
                                          baseline_time=(-0.25, 0)):
    '''
    Compute a response contrast but control for
    mean contrast level. Do this by splitting up
    trials into several contrast groups and then computing
    individual response contrasts within.
    '''
    meta = preprocessing.get_meta_for_subject(sub, 'stimulus')

    # Compute Hits, Miss, CR and FA
    # Positive -> Yes, Negative -> No
    # [-2, 2]-> correct, [-1, 1] -> err:
    response_type = ((1 + meta.correct) * meta.response).values
    hits = response_type == 2
    cr = response_type == -2
    miss = response_type == -1
    fa = response_type == 1

    filter_dict = {
        'HIT': meta.reset_index().loc[hits, 'hash'].values,
        'FA': meta.reset_index().loc[fa, 'hash'].values,
        'CR': meta.reset_index().loc[cr, 'hash'].values,
        'MISS': meta.reset_index().loc[miss, 'hash'].values
    }

    if sub <= 8:
        hand_mapping = {'CR': 'lh_is_ipsi',
                        'MISS': 'lh_is_ipsi',
                        'HIT': 'rh_is_ipsi',
                        'FA': 'rh_is_ipsi'}
    else:
        hand_mapping = {'HIT': 'lh_is_ipsi',
                        'FA': 'lh_is_ipsi',
                        'CR': 'rh_is_ipsi',
                        'MISS': 'rh_is_ipsi'}
    return contrast(sub, filter_dict, hand_mapping, None,
                    epoch=epoch, baseline_time=baseline_time)


def response_contrast(subs=list(range(1, 16)), epoch='stimulus'):
    return transcode_columns(pd.concat(
        [_prewarm_response_contrast(sub, epoch=epoch) for sub in subs]))


def submit_response_contrast():
    from pymeg import parallel
    from itertools import product
    for subject, epoch in product(range(1, 16),  ['stimulus', 'response']):

        parallel.pmap(
            _prewarm_response_contrast, [(subject, epoch)],
            walltime=10, memory=40, nodes=1, tasks=1,
            name='PRWRC' + str(subject) + epoch,
            ssh_to=None)


@memory.cache
def _prewarm_response_contrast(sub, epoch='stimulus', baseline_time=(-0.25, 0)):
    meta = preprocessing.get_meta_for_subject(sub, 'stimulus')
    filter_dict = {'M1': meta.query('response==-1').reset_index().loc[:, 'hash'].values,
                   'P1': meta.query('response==1').reset_index().loc[:, 'hash'].values}
    if sub <= 8:
        hand_mapping = {'M1': 'lh_is_ipsi', 'P1': 'rh_is_ipsi'}
    else:
        hand_mapping = {'M1': 'rh_is_ipsi', 'P1': 'lh_is_ipsi'}
    return contrast(sub, filter_dict, hand_mapping, ['P1', 'M1'],
                    epoch=epoch, baseline_time=baseline_time)


def side_contrast(subs=range(1, 16), epoch='stimulus'):
    return pd.concat(
        [_prewarm_side_contrast(sub, epoch=epoch) for sub in subs])


def _prewarm_side_contrast(sub, epoch='stimulus', baseline_time=(-0.25, 0)):
    meta = preprocessing.get_meta_for_subject(sub, 'stimulus')
    filter_dict = {'Spos': meta.query('side==-1').reset_index().loc[:, 'hash'].values,
                   'Sneg': meta.query('side==1').reset_index().loc[:, 'hash'].values}
    if sub <= 8:
        hand_mapping = {'Spos': 'lh_is_ipsi', 'Sneg': 'rh_is_ipsi'}
    else:
        hand_mapping = {'Sneg': 'rh_is_ipsi', 'Spos': 'lh_is_ipsi'}
    return contrast(sub, filter_dict, hand_mapping, ['P1', 'M1'],
                    epoch=epoch, baseline_time=baseline_time)


def contrast(sub, filter_dict, hand_mapping, contrast,
             epoch='stimulus', baseline_epoch='stimulus',
             baseline_time=(-0.25, 0)):
    tfrs = []

    trial_list = list(filter_dict.values())
    condition_names = list(filter_dict.keys())
    condition_tfrs, weights = load_sub_grouped_weighted(
        sub, trial_list, epoch=epoch)

    if baseline_epoch == epoch:
        baseline_tfrs = [t.copy() for t in condition_tfrs]
    else:
        baseline_tfrs, weights = load_sub_grouped_weighted(
            sub, trial_list, epoch=baseline_epoch)

    for condition, tfr, base_tfr in zip(condition_names, condition_tfrs, baseline_tfrs):
        tfr.loc[:, 'condition'] = condition
        # tfr.set_index('condition', append=True, inplace=True)
        base_tfr.loc[:, 'condition'] = condition
        # base_tfr.set_index('condition', append=True, inplace=True)
        # Baseline correct here
        tfr = tfr.reset_index().set_index(
            ['sub', 'est_key', 'est_val', 'condition', 'time'])
        base_tfr.set_index('condition', append=True, inplace=True)
        groups = []
        base_tfr = base_tfr.reset_index().set_index(
            ['sub', 'est_key', 'est_val', 'condition', 'time'])
        for gp, group in tfr.groupby(['sub', 'est_key',
                                      'est_val', 'condition']):
            base = base_tfr.loc[gp, :]
            group = baseline(group, base, baseline=baseline_time)
            groups.append(group)

        tfr = pd.concat(groups)
        # left, right = rois.lh(tfr.columns), rois.rh(tfr.columns)
        left, right = sorted(rois.lh(tfr.columns)), sorted(
            rois.rh(tfr.columns))
        if len(left) < len(right):
            right = sorted(l.replace('lh', 'rh') for l in left)
        elif len(left) > len(right):
            left = sorted(l.replace('rh', 'lh') for l in right)

        # Now compute lateralisation
        if hand_mapping is not None:
            if hand_mapping[condition] == 'lh_is_ipsi':
                print(sub, condition, 'Left=IPSI')
                ipsi, contra = left, right
            elif hand_mapping[condition] == 'rh_is_ipsi':
                print(sub, condition, 'Right=IPSI')
                ipsi, contra = right, left
            else:
                raise RuntimeError('Do not understand hand mapping')
            lateralized = rois.lateralize(tfr, ipsi, contra)

        # Averge hemispheres
        left, right = sorted(rois.lh(tfr.columns)), sorted(
            rois.rh(tfr.columns))
        if len(left) > len(right):
            right = sorted(l.replace('lh', 'rh') for l in left)
        elif len(left) < len(right):
            left = sorted(l.replace('rh', 'lh') for l in right)
        havg = pd.concat(
            [tfr.loc[:, (x, y)].mean(1) for x, y in zip(left, right)],
            1)
        havg.columns = [x + '_Havg' for x in left]

        tfrs.append(pd.concat([tfr, lateralized, havg], 1))
    tfrs = pd.concat(tfrs, 0)
    # cond1 = tfrs.query('condition=="%s"' % contrast[0])
    # cond2 = tfrs.query('condition=="%s"' % contrast[1])
    # delta = (cond1.reset_index(level='condition', drop=True) -
    #         cond2.reset_index(level='condition', drop=True))
    # delta.loc[:, 'condition'] = 'diff'
    # delta.set_index('condition', append=True, inplace=True)
    return tfrs  # pd.concat([tfrs, delta], )


def load_sub_grouped_weighted(sub, trials, epoch='stimulus'):
    '''
    Load average TFR values generated from a specific
    set of trials.

    Trials from different sessions are loaded and averaged per
    session. Session averages are computed by weighting each
    session with its fraction of trials.

    Parameters:
        sub : int, subject number
        trials : list of trial hashes
    '''
    sacc = []
    stdacc = []
    weights = []
    for session in range(4):
        try:
            df, stds, ws = load_sub_session_grouped(
                sub, session, trials, epoch=epoch)
            sacc.append(df)
            stdacc.append(stds)
            weights.append(ws)
        except IOError:
            print('No data for %i ,%i' % (sub, session))
    n_conditions = len(sacc[0])
    conditions = []
    stds = []

    for cond in range(n_conditions):
        averages = pd.concat([s[cond] for s in sacc]).groupby(
            ['sub', 'time', 'est_key', 'est_val']).mean()
        conditions.append(averages)

    return conditions, weights


@memory.cache
def load_sub_session_grouped(sub, session, trial_list, epoch='stimulus',
                             baseline=(-0.25, 0), log10=True):

    if epoch == 'stimulus':
        filenames = glob(
            '/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-stimulus-F-chunk*-lcmv.hdf' % (
                sub, session))
    elif epoch == 'response':
        filenames = glob(
            '/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-response-F-chunk*-lcmv.hdf' % (
                sub, session))
    else:
        raise RuntimeError('Do not understand epoch %s' % epoch)
    df = pd.concat([pd.read_hdf(f) for f in filenames])
    # Has index trial time est_key est_val
    # df.set_index(['trial', 'time',
    #              'est_key', 'est_val'], inplace=True)
    if log10:
        df = np.log10(df) * 10

    df = rois.reduce(df).reset_index()  # Reduce to visual clusters

    #all_trials = df.loc[:, 'trial']
    conditions = []
    weights = []
    stds = []
    df.set_index('trial', inplace=True)
    total_trials = len(np.unique(df.index.values))
    for trials in trial_list:
        id_trials = np.unique(df.index.intersection(trials))
        n_trials = len(np.unique(id_trials))
        # df_sub = df.set_index('trial').loc[trials, :].reset_index()
        df_sub = df.loc[id_trials].reset_index()
        weights.append(float(n_trials) / total_trials)
        df_sub.loc[:, 'sub'] = sub
        df_sub.loc[:, 'session'] = session

        conditions.append(df_sub.groupby(
            ['session', 'sub', 'time', 'est_key', 'est_val']).mean())

        stds.append((
            df_sub.query('%f < time & time < %f' % baseline)
                  .groupby(['session', 'sub', 'est_key', 'est_val'])
                  .std()))
    return conditions, stds, weights


@memory.cache
def stats_test(data, n_permutations=1000):
    from mne.stats import permutation_cluster_1samp_test
    threshold_tfce = dict(start=0, step=0.2)
    t_tfce, _, p_tfce, H0 = permutation_cluster_1samp_test(
        data.swapaxes(1, 2), n_jobs=4, threshold=threshold_tfce, connectivity=None, tail=0,
        n_permutations=n_permutations)
    return t_tfce.T, p_tfce.reshape(t_tfce.shape).T, H0


def get_tfr_stack(data, area, baseline=None, tslice=slice(-0.25, 1.35)):
    stack = []
    for sub, ds in data.groupby('sub'):
        try:
            stack.append(get_tfr(ds, area, tslice=tslice).values)
        except ValueError:
            print('No data fpr sub %i, area %s' % (sub, area))

    return np.stack(stack)


def get_tfr(data, area, baseline=None, tslice=slice(-0.25, 1.35)):
    data = data.loc[:, area].dropna()
    if len(data) == 0:
        raise ValueError('No data for this %s' % area)
    k = pd.pivot_table(data.reset_index(), values=area,
                       index='est_val', columns='time')
    if baseline is not None:
        k = k.subtract(k.loc[:, baseline].mean(1), axis=0).div(
            k.loc[:, baseline].std(1), axis=0)
    return k.loc[:, tslice]

'''
The following functions allow plotting of TFRs.
'''


def plot_set(response, stimulus, setname, setareas, minmax=(10, 20),
             stats=False,
             response_tslice=slice(-1, 0.5),
             stimulus_tslice=slice(-0.25, 1.35),
             new_figure=True):
    from matplotlib.gridspec import GridSpec
    import pylab as plt

    columns = response.columns
    areas = rois.filter_cols(columns, setareas)
    # Setup gridspec to compare stimulus and response next to each other.    
    cols = 3
    rows = (len(areas)//cols)+1

    if new_figure:
        plt.figure(figsize=(cols * 2.5, rows * 2.5))

    gs = GridSpec(2 * rows, 2 * cols, height_ratios=[140, 16] * rows,
                  width_ratios=[1.55, 1.5] * cols)
    locations = []

    # First plot stimulus and comput stimulus positions in plot.
    for ii, area in enumerate(setareas):
        row, col = (ii // cols) * 2, np.mod(ii, cols)
        locations.append((row, col * 2))
    plot_labels(stimulus, areas, locations, gs,
                minmax=minmax, stats=stats, tslice=stimulus_tslice)

    # First plot stimulus and comput stimulus positions in plot.
    locations = [(row, col + 1) for row, col in locations]

    plot_labels(response, areas, locations, gs,
                minmax=minmax, stats=stats, tslice=response_tslice)


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
        #ex_tfr = get_tfr(data.query('est_key=="LF"'), area, tslice=tslice)
        #s = get_tfr_stack(data.query('est_key=="LF"'), area, tslice=tslice)
        #if stats:
        #    t, p, H0 = stats_test(s)
        #    p = p.reshape(t.shape)
        #cbar = _plot_tfr(area, ex_tfr.columns.values, ex_tfr.index.values,
        #                 s.mean(0), p, title_color='k', minmax=minmax[1])
        #cbar.remove()
        # plt.xticks([0, 0.5, 1])
        if row == maxrow:
            plt.xlabel('time')

            # plt.xticks([tslice.start, 0, tslice.stop])
        else:
            plt.xticks([])


def plot_tfr(data, area, ps=None, minmax=None, title_color='k'):
    tfr = get_tfr(data, area)
    tfr_values = get_tfr_stack(data, area).mean(0)
    _plot_tfr(area, tfr.columns.values, tfr.index.values,
              tfr_values, ps, title_color=title_color, minmax=minmax)


def _plot_tfr(area, columns, index, tfr_values, ps, title_color='k',
              minmax=None):
    import pylab as plt
    import seaborn as sns

    if minmax is None:
        minmax = np.abs(np.percentile(tfr_values, [1, 99])).max()
    di = np.diff(index)[0] / 2.
    plt.imshow(np.flipud(tfr_values), cmap='RdBu_r', vmin=-minmax, vmax=minmax,
               extent=[columns[0], columns[-1],
                       index[0] - di, index[-1] + di], aspect='auto',
               interpolation='none')

    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.5))
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    sns.despine(ax=plt.gca(), bottom=True)
    cbar = plt.colorbar()
    cbar.set_ticks([np.ceil(-minmax), np.floor(minmax)])
    if ps is not None:
        plt.contour(ps, [0.05], extent=[columns[0], columns[-1], index[0] - di, index[-1] + di],
                    origin='lower')

    tarea = (area
             .replace('lh.JWDG.', '')
             .replace('lh.a2009s.', '')
             .replace('rh.JWDG.', '')
             .replace('rh.a2009s.', '')
             .replace('lh.wang2015atlas.', '')
             .replace('rh.wang2015atlas.', '')
             .replace('-lh_Lateralized', '')
             .replace('-lh_Havg', ''))

    plt.xlim(columns[0], columns[-1])
    plt.xticks([0])
    if index.max() > 30:
        plt.title(tarea, color=title_color)
        plt.yticks([10,  50, 100,  150])
        plt.ylim([10 - di, 150 + di])
    else:
        plt.yticks([4, 20])
        plt.ylim([4 - di, 20 + di])
    return cbar


def make_tableau(data, areas={'lh': 'M1-lh', 'rh': 'M1-rh'}, **kwargs):
    import pylab as plt
    for i, (area, condition) in enumerate(zip(['lh', 'rh', 'lh', 'rh'],
                                              ['P1', 'P1', 'M1', 'M1'])):
        plt.subplot(2, 2, i + 1)
        tfr = get_tfr(data.query('condition=="%s"' %
                                 condition), areas[area],  tslice=slice(-1, 0.5))
        _plot_tfr(areas[area], tfr.columns, tfr.index,
                  tfr.values, None, **kwargs)
