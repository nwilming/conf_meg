import os
import pandas as pd
from conf_analysis.meg import preprocessing
from pymeg.contrast_tfr import Cache, compute_contrast, augment_data
from pymeg import contrast_tfr
from pymeg import parallel
from joblib import Memory
import logging

logging.getLogger().setLevel(logging.INFO)
logger = contrast_tfr.logging.getLogger()
logger.setLevel(logging.INFO)


if 'TMPDIR' in os.environ.keys():
    data_path = '/nfs/nwilming/MEG/'
else:
    data_path = '/home/nwilming/conf_meg/'

memory = Memory(location=os.environ['PYMEG_CACHE_DIR'], verbose=0)


contrasts = {
    'all': (['all'], [1]),
    'choice': (['hit', 'fa', 'miss', 'cr'], (1, 1, -1, -1)),
    'stimulus': (['hit', 'fa', 'miss', 'cr'], (1, -1, 1, -1)),
    'hand': (['left', 'right'], (0.5, -0.5)),
    'confidence': (['high_conf_high_contrast', 'high_conf_low_contrast',
                    'low_conf_high_contrast', 'low_conf_low_contrast'],
                   (0.5, -0.5, 0.5, -0.5)),  # (HCHC-HCLC) + (LCHC - LCLC)
    'confidence_asym': (['high_conf_high_contrast', 'high_conf_low_contrast',
                         'low_conf_high_contrast', 'low_conf_low_contrast'],
                        (0.5, -0.5, -0.5, +0.5)),
    # (HCHC-HCLC) - (LCHC - LCLC)
}


def submit_contrasts(collect=False):
    tasks = []
    for subject in [3]:
        # for session in range(0, 4):
        tasks.append((contrasts,  subject))
    res = []
    for task in tasks:
        try:
            r = _eval(get_contrasts, task, collect=collect,
                      walltime='01:30:00', tasks=4, memory=60)
            res.append(r)
        except RuntimeError:
            print('Task', task, ' not available yet')
    return res


def _eval(func, args, collect=False, **kw):
    """
    Intermediate helper to toggle cluster vs non cluster
    """
    if not collect:
        if not func.in_store(*args):
            print('Submitting %s to %s for parallel execution' %
                  (str(args), func))
            parallel.pmap(func, [args], **kw)
    else:
        if func.in_store(*args):
            print('Submitting %s to %s for collection' % (str(args), func))
            df = func(*args)
            return df
        else:
            raise RuntimeError('Result not available.')


@memory.cache(ignore=['scratch'])
def get_contrasts(contrasts, subject, baseline_per_condition=False,
                  scratch=False):

    if subject < 8:
        hemi = 'lh_is_ipsi'
    else:
        hemi = 'rh_is_ipsi'
    hemis = [hemi, 'avg']
    from os.path import join
    stim, resp = [], []
    for session in range(0, 4):
        stim.append(join(data_path, 'sr_labeled/S%i-SESS%i-stimulus*.hdf' % (
            subject, session)))
        resp.append(join(data_path, 'sr_labeled/S%i-SESS%i-response*.hdf' % (
            subject, session)))

    if scratch:
        from subprocess import run
        import os
        tmpdir = os.environ['TMPDIR']
        command = ('cp {stim} {tmpdir} & cp {resp} {tmpdir}'
                   .format(stim=stim, resp=resp, tmpdir=tmpdir))
        logging.info('Copying data with following command: %s' % command)
        p = run(command, shell=True, check=True)
        stim = join(data_path, tmpdir, 'S%i-SESS%i-stimulus*.hdf' % (
            subject, session))
        resp = join(data_path, tmpdir, 'S%i-SESS%i-response*.hdf' % (
            subject, session))
        logging.info('Copied data')

    meta = preprocessing.get_meta_for_subject(subject, 'stimulus')
    response_left = meta.response == 1
    stimulus = meta.side == 1
    meta = augment_data(meta, response_left, stimulus)
    meta["high_conf_high_contrast"] = (meta.confidence == 2) & (meta.mc > 0.5)
    meta["high_conf_low_contrast"] = (meta.confidence == 1) & (meta.mc > 0.5)
    meta["low_conf_high_contrast"] = (meta.confidence == 2) & (meta.mc <= 0.5)
    meta["low_conf_low_contrast"] = (meta.confidence == 1) & (meta.mc <= 0.5)
    cps = []
    with Cache() as cache:
        try:
            contrast = compute_contrast(
                contrasts, hemis, stim, stim,
                meta, (-0.25, 0),
                baseline_per_condition=baseline_per_condition,
                n_jobs=1, cache=cache)
            contrast.loc[:, 'epoch'] = 'stimulus'
            cps.append(contrast)
        except ValueError as e: 
            # No objects to concatenate
            print(e)
            pass
        try:
            contrast = compute_contrast(
                contrasts, hemis, resp, stim,
                meta, (-0.25, 0),
                baseline_per_condition=baseline_per_condition,
                n_jobs=1, cache=cache)
            contrast.loc[:, 'epoch'] = 'response'
            cps.append(contrast)
        except ValueError as e:
            # No objects to concatenate
            print(e)
            pass
    contrast = pd.concat(cps)
    del cps
    contrast.loc[:, 'subject'] = subject

    contrast.set_index(['subject',  'contrast',
                        'hemi', 'epoch', 'cluster'], append=True, inplace=True)
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
