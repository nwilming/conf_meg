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
    'hand': (['left', 'right'], (1, -1)),
    'confidence': (
        ['high_conf_high_contrast', 'high_conf_low_contrast',
         'low_conf_high_contrast', 'low_conf_low_contrast'],
        (1, 1, -1, -1)),
    # (HCHCont - LCHCont) + (HCLCont - LCLCont) ->
    #  HCHCont + HCLcont  -  LCHCont - LCLCont
    # Example: 
    # HCm = 10, LCm = 7 (10-7) + (1-4) = 3-3 == 0

    'confidence_asym': (
        ['high_conf_high_contrast', 'high_conf_low_contrast',
         'low_conf_high_contrast', 'low_conf_low_contrast'],
        (1, -1, -1, 1)),
    # (HCHCont - LCHCont) - (HCLCont - LCLCont) ->
    #  HCHCont - HCLcont  -  LCHCont + LCLCont
}


def submit_contrasts(collect=False):
    tasks = []
    for subject in range(1, 16):
        # for session in range(0, 4):
        tasks.append((contrasts,  subject))
    res = []
    for task in tasks:
        try:
            r = _eval(get_contrasts, task, collect=collect,
                      walltime='01:30:00', tasks=5, memory=70)
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


def plot_mosaics(df, stats=False):
    import pylab as plt
    from pymeg.contrast_tfr import plot_mosaic
    for epoch in ['stimulus', 'response']:
        for contrast in ['all', 'choice', 'confidence', 'confidence_asym', 'hand', 'stimulus']:
            for hemi in [True, False]:
                plt.figure()
                query = 'epoch=="%s" & contrast=="%s" & %s(hemi=="avg")' % (
                    epoch, contrast, {True: '~', False: ''}[hemi])
                d = df.query(query)
                plot_mosaic(d, epoch=epoch, stats=stats)
                plt.suptitle(query)
                plt.savefig(
                    '/Users/nwilming/Desktop/tfr_average_%s_%s_lat%s.pdf' % (epoch, contrast, hemi))
                plt.savefig(
                    '/Users/nwilming/Desktop/tfr_average_%s_%s_lat%s.svg' % (epoch, contrast, hemi))
# Ignore following for now


def submit_stats(
        contrasts=['all', 'choice', 'confidence',
                   'confidence_asym', 'hand',
                   'stimulus'],
        collect=False):
    all_stats = {}
    tasks = []
    for contrast in contrasts:
        for hemi in [True, False]:
            for epoch in ['stimulus', 'response']:
                tasks.append((contrast, epoch, hemi))
    res = []
    for task in tasks[:]:
        try:
            r = _eval(precompute_stats, task, collect=collect,
                      walltime='08:30:00', tasks=2, memory=20)
            res.append(r)
        except RuntimeError:
            print('Task', task, ' not available yet')
    return res


@memory.cache()
def precompute_stats(contrast, epoch, hemi):
    from pymeg import atlas_glasser
    df = pd.read_hdf('/home/nwilming/all_contrasts_confmeg-20190108.hdf')
    if epoch == "stimulus":
        time_cutoff = (-0.5, 1.35)
    else:
        time_cutoff = (-1, .5)
    query = 'epoch=="%s" & contrast=="%s" & %s(hemi=="avg")' % (
        epoch, contrast, {True: '~', False: ''}[hemi])
    df = df.query(query)
    all_stats = {}
    for (name, area) in atlas_glasser.areas.items():
        task = contrast_tfr.get_tfr(
            df.query('cluster=="%s"' % area), time_cutoff)
        all_stats.update(contrast_tfr.par_stats(*task, n_jobs=1))
    return all_stats


def plot_2epoch_mosaics(df, stats=False, contrasts=['all', 'choice', 'confidence', 'confidence_asym', 'hand', 'stimulus']):
    import pylab as plt
    from pymeg.contrast_tfr import plot_2epoch_mosaic

    for contrast in contrasts:
        for hemi in [True, False]:
            plt.figure()
            query = 'contrast=="%s" & %s(hemi=="avg")' % (
                contrast, {True: '~', False: ''}[hemi])
            d = df.query(query)
            plot_2epoch_mosaic(d, stats=stats, cmap='RdBu')
            title = '%s, %s' % (
                contrast, {True: 'Lateralized', False: 'Hemis avg.'}[hemi])
            plt.suptitle(title)
            plt.savefig(
                '/Users/nwilming/Desktop/tfr_average_2e_%s_lat%s.pdf' % (
                    contrast, hemi),
                bbox_inches='tight')
        



            
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
