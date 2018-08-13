#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Decoding analyses for conf_meg data.

4.  Signed confidence and choice decoder: Same as MIDC and MDDC but with
    confidence folded into the responses (-2, -1, 1, 2)
5.  Unsigned confidence decoder: Same as MIDC and MDDC but decode
    confidence only.
'''

import logging
import os
import numpy as np
import pandas as pd

from functools import partial
from itertools import product

from sklearn import linear_model, discriminant_analysis, svm
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing

from pymeg import roi_clusters as rois

from joblib import Memory

if 'RRZ_LOCAL_TMPDIR' in os.environ.keys():
    memory = Memory(cachedir=os.environ['RRZ_LOCAL_TMPDIR'])
if 'TMPDIR' in os.environ.keys():
    tmpdir = os.environ['TMPDIR']
    memory = Memory(cachedir=tmpdir)
else:
    memory = Memory(cachedir=metadata.cachedir)

areas = [
    'vfcvisual', 'vfcVO', 'vfcPHC', 'vfcV3ab',
    'vfcTO', 'vfcLO', 'vfcIPS_occ', 'vfcIPS_dorsal',
    'vfcSPL',  'vfcFEF', 'IPS_Pces', 'M1', 'aIPS1',
    'LIP', 'Area6_dorsal_medial', 'Area6_anterior',
    'A10', 'A6si', 'PEF', '55b', '8Av',
    '8Ad', '9p', '8Bl', '8C', 'p9-46v', '46',
    '9-46d', 'a10p', 'a47r', 'p32', 's32', 'a24',
    '9m', 'd32', 'a32pr', 'p32pr', '24dv', 'p24pr',
]

avgs = [x + '-lh_Havg' for x in areas]
lateralized = [x + '-lh_L-R' for x in areas]
pairs = [(x + '-lh', x + '-rh') for x in areas]

areas = avgs + lateralized + pairs
n_jobs = 1

n_trials = {1: {'stimulus': 1565, 'response': 245},
            2: {'stimulus': 1852, 'response': 1697},
            3: {'stimulus': 1725, 'response': 27},
            4: {'stimulus': 1863, 'response': 1807},
            5: {'stimulus': 1812, 'response': 877},
            6: {'stimulus': 1128, 'response': 113},
            7: {'stimulus': 1766, 'response': 1644},
            8: {'stimulus': 1872, 'response': 1303},
            9: {'stimulus': 1767, 'response': 1104},
            10: {'stimulus': 1404, 'response': 1209},
            11: {'stimulus': 1787, 'response': 1595},
            12: {'stimulus': 1810, 'response': 1664},
            13: {'stimulus': 1689, 'response': 1620},
            14: {'stimulus': 1822, 'response': 1526},
            15: {'stimulus': 1851, 'response': 1764}}


def submit(cluster='PBS'):
    # decoders = ['MIDC_split', 'MIDC_nosplit',
    #            'SIDE_nosplit',
    #            'CONF_signed', 'CONF_unsigned']
    #decoders = ['CONF_unsign_split']
    decoders = ['SSD_delta_contrast',
                'SSD_acc_contrast',
                'SSD_acc_contrast_diff']
    from pymeg import parallel
    if cluster == 'slurm':
        pmap = partial(parallel.pmap, email=None, tasks=1, nodes=1, memory=60,
                       ssh_to=None, home='/work/faty014', walltime='11:50:55',
                       cluster='SLURM')
    else:
        pmap = partial(parallel.pmap, nodes=1, tasks=1, memory=31,
                       ssh_to=None,  walltime=48, env='py36')

    for subject, epoch, decoder in product(range(1, 16),
                                           ['stimulus', 'response'],
                                           decoders):
        pmap(run_AA, [(subject, decoder, epoch)],
             name='DCD' + decoder + epoch + str(subject),
             )

    # for subject in range(1, 16):
    #    pmap(run_AA, [(subject, 'SSD', 'stimulus')],
    #         name='DCDSSDStimulus' + str(subject))


def run_AA(subject, decoder, epoch, ntasks=16):
    from multiprocessing import Pool
    p = Pool(ntasks)
    print('Starting pool of workers')
    latencies = {'stimulus': [-0.5, 1.35], 'response': [-1, 0.5]}
    time = latencies[epoch]
    est_vals = (10, 150)

    data = pd.concat([
        get_all_areas(subject, session, epoch=epoch, time=(-1.5, 1),
                      est_vals=(10, 150), est_key='F',
                      baseline_epoch='stimulus', baseline=(-0.35, 0))
        for session in range(4)])
    tasks = []
    for area in areas:
        # subject, area, epoch='stimulus', time=(0, 1),
        # est_vals=(30, 100), est_key='F',
        # baseline_epoch='stimulus', baseline=(-0.35, 0),
        # data=None
        get_subject(subject, area, epoch=epoch, time=time,
                    est_vals=est_vals, data=data)
    # p.starmap(get_subject, tasks, chunksize=ntasks)
    del data
    print('Done caching')

    args = []
    for est_key, est_vals in zip(['F'], [est_vals]):
        for area in areas:
            args.append((subject, decoder, epoch, area, est_key, est_vals))
    scores = p.starmap(run_single_area, args, chunksize=ntasks)
    scores = [s for s in scores if s is not None]
    scores = pd.concat(scores)

    if (('RRZ_TMPDIR' in list(os.environ.keys()))
            or ('RRZ_LOCAL_TMPDIR' in list(os.environ.keys()))):
        outpath = '/work/faty014/MEG/sr_decoding/'
    else:
        outpath = '/nfs/nwilming/MEG/sr_decoding/'
    filename = outpath + '/concat_S%i-%s-%s-decoding.hdf' % (
        subject, decoder, epoch)
    scores.to_hdf(filename, 'decoding')
    p.terminate()
    return scores


def run_single_area(subject, decoder, epoch, area, est_key='F', est_vals=[10, 150],
                    ignore_errors=False):
    try:
        return run_decoder(subject, decoder, area, epoch)
    except:
        print('Error in runAA: (%i, %s, %s, %s)' %
              (subject, decoder, epoch, area))
        if not ignore_errors:
            raise


def run_decoder(subject, decoder, area, epoch, est_key='F', est_vals=[10, 150]):
    '''
    Run a specific type decoding on a subject's data set

    Arguments
    ---------

    subject : int, subject number
    decoder : str
        Selects different decoding targets. One of
            SSD - Decode sample contrast (regression)
            MIDC - Decode response (classification)
            SCONF - Decode signed confidence (classification)
            UCONF - Decode unsigned confidence (classification)
    area : str
        Selects which brain area to use
    epoch : str, 'stimulus' or 'response'
    '''
    def get_save_path(subject, decoder, area, epoch):
        import socket
        if (('RRZ_TMPDIR' in list(os.environ.keys())) or
                ('RRZ_LOCAL_TMPDIR' in list(os.environ.keys()))):
            path = '/work/faty014/MEG/'

        elif 'lisa.surfsara' in socket.gethostname():
            path = '/nfs/nwilming/MEG/'
        else:
            path = '/home/nwilming/conf_meg/'
        filename = 'sr_decoding/S%i-%s-%s-%s-decoding.hdf' % (
            subject, decoder, epoch, area)
        return os.path.join(path, filename)

    filename = get_save_path(subject, decoder, epoch, area)

    try:
        k = pd.read_hdf(filename)
        logging.info('Loaded cached file: %s' % filename)
        return k
    except IOError:
        pass

    latencies = {'stimulus': [-0.5, 1.35], 'response': [-1, 0.5]}

    data, meta = get_subject(subject, epoch=epoch,
                             area=area, time=latencies[epoch],
                             est_vals=est_vals, est_key=est_key)
    data.reset_index().set_index('trial', inplace=True)

    dt = np.diff(data.time)[0]
    latencies = np.unique(data.time)
    meta.loc[:, 'signed_confidence'] = (meta.loc[:, 'confidence'] *
                                        meta.loc[:, 'response']).astype(int)
    meta.loc[:, 'unsigned_confidence'] = (
        meta.loc[:, 'confidence'] == 1).astype(int)

    decoders = {'SSD': ssd_decoder,
                'SSD_delta_contrast': partial(
                    ssd_decoder, target_value='delta_contrast'),
                'SSD_acc_contrast': partial(
                    ssd_decoder, target_value='acc_contrast'),
                'SSD_acc_contrast_diff': partial(
                    ssd_decoder, target_value='acc_contrast_diff'),
                'MIDC_split': partial(midc_decoder, splitmc=True,
                                      target_col='response'),
                'MIDC_nosplit': partial(midc_decoder, splitmc=False,
                                        target_col='response'),
                'SIDE_split': partial(midc_decoder, splitmc=True,
                                      target_col='side'),
                'SIDE_nosplit': partial(midc_decoder, splitmc=False,
                                        target_col='side'),
                'CONF_signed': partial(midc_decoder, splitmc=False,
                                       target_col='signed_confidence'),
                'CONF_unsigned': partial(midc_decoder, splitmc=False,
                                         target_col='unsigned_confidence'),
                'CONF_unsign_split': partial(midc_decoder, splitmc=True,
                                             target_col='unsigned_confidence')}

    if decoder == 'SSD':
        latencies = np.arange(-.1, 0.75, dt)

    assert(decoder in list(decoders.keys()))
    scores = []
    for latency in latencies:
        logging.info('S=%i; DCD=%s, EPOCH=%s; t=%f' %
                     (subject, decoder, epoch, latency))
        try:
            s = decoders[decoder](meta, data, area, latency=latency)
            scores.append(s)
        except:
            logging.exception(''''Error in run_decoder:
        # Subject: %i
        # Decoder: %s
        # Epoch: %s
        # Area: %s
        # Latency: %f)''' % (subject, decoder, epoch, area, latency))
            raise
    scores = pd.concat(scores)
    scores.loc[:, 'signal'] = decoder
    scores.loc[:, 'subject'] = subject
    scores.loc[:, 'epoch'] = epoch
    scores.loc[:, 'area'] = str(area)
    scores.loc[:, 'est_key'] = est_key
    scores.to_hdf(filename, 'decoding')
    return scores


def ssd_decoder(meta, data, area, latency=0.18, target_value='contrast'):
    '''
    Sensory signal decoder (SSD).

    Tries to decode the contrast value of individual samples
    within the trial. The decoder used for this will be some form
    of regression based approach. Input signals will be pooled
    across hemispheres.
    '''

    # Build a design matrix:
    # Each sample is one observation
    # Each frequency is one feature
    # Contrast is the target

    # Map sample times to existing time points in data
    from sklearn.metrics import make_scorer
    from scipy.stats import linregress
    slope_scorer = make_scorer(lambda x, y: linregress(x, y)[0])
    corr_scorer = make_scorer(lambda x, y: np.corrcoef(x, y)[0, 1])
    sample_scores = []
    for sample_num, sample in enumerate(np.arange(0, 1, 0.1)):
        target_time_points = [sample + latency]
        times = np.unique(data.reset_index().time)
        target_time_points = np.array(
            [times[np.argmin(abs(times - t))] for t in target_time_points]
        )
        # Compute mean latency
        latency = np.mean(target_time_points - sample)
        # print('Latency:', latency, '->', target_time_points)
        # Turn data into (trial X Frequency)
        X = []
        for a in ensure_iter(area):
            x = (data.reset_index()
                 .loc[:, ['trial', 'time', 'est_val', a]]
                 .set_index(['trial', 'time', 'est_val'])
                 .unstack())
            X.append(x)
        sample_data = pd.concat(X, 1)
        sample_data = sample_data.loc[(slice(None), target_time_points), :]

        # Build target vector
        cvals = np.stack(meta.contrast_probe.values)
        if target_value == 'contrast':
            target = cvals[:, sample_num]
        elif target_value == 'delta_contrast':
            if sample_num == 0:
                target = cvals[:, sample_num]
            else:
                target = cvals[:, sample_num] - cvals[:, sample_num - 1]
        elif target_value == 'acc_contrast':
            target = cvals[:, :(sample_num + 1)].mean(1)
        elif target_value == 'acc_contrast_diff':
            if sample_num == 0:
                target = cvals[:, sample_num]
            else:
                target = cvals[:, sample_num] - \
                    cvals[:, :(sample_num)].mean(1)
        else:
            raise RuntimeError('Do not understand target: %s' % target)

        metrics = {'explained_variance': 'explained_variance',
                   'r2': 'r2',
                   'slope': slope_scorer,
                   'corr': corr_scorer}

        classifiers = {
            'OLS': linear_model.LinearRegression(),
            'Ridge': linear_model.RidgeCV()}

        for name, clf in list(classifiers.items()):
            clf = Pipeline([
                ('Scaling', StandardScaler()),
                (name, clf)
            ])
            score = cross_validate(clf, sample_data.values, target,
                                   cv=10, scoring=metrics,
                                   return_train_score=False,
                                   n_jobs=n_jobs)
            # fit = clf(sample_data.values, target)
            del score['fit_time']
            del score['score_time']
            score = {k: np.mean(v) for k, v in list(score.items())}
            score['latency'] = latency
            score['Classifier'] = name
            score['sample'] = sample_num
            # score['coefs'] = fit.coef_.astype(object)
            sample_scores.append(score)

    return pd.DataFrame(sample_scores)


def midc_decoder(meta, data, area, latency=0, splitmc=True,
                 target_col='response'):
    '''
    This signal decodes choices of participants based on local
    reconstructed brain activity. The decoding needs to be decoupled
    from evidence strength. To this end decoding will be carried out
    in several bins that all have the same mean contrast.

    This will be somewhat challenging because for
    high mean contrast behavior will obviously be very different, so
    very uneven class sizes. I'd use overlapping bins of equal size here.
    An interesting question would be to see if the decoder transfers
    between bins, this would be the strongest test of evidence
    independent decoding.

    Set the input data frame and area to 'Havg' for motor independent
    signals, and to 'Lateralized' for motor dependent choice signals
    (MDDC).
    '''

    # Map to nearest time point in data
    times = np.unique(data.reset_index().time)
    target_time_point = times[np.argmin(abs(times - latency))]
    logging.info('Selecting next available time point: %02.2f' %
                 target_time_point)
    times_idx = np.isclose(data.time, target_time_point, 0.00001)
    data = data.loc[times_idx, :]

    # Turn data into (trial X Frequency)
    X = []
    for a in ensure_iter(area):
        x = (data.reset_index()
             .loc[:, ['trial', 'est_val', a]]
             .set_index(['trial', 'est_val'])
             .unstack())
        X.append(x)
    data = pd.concat(X, 1)
    scores = []
    if splitmc:
        for mc, sub_meta in meta.groupby(meta.mc < 0.5):
            # Align data and meta
            # sub_meta = meta
            # sub_data = data

            sub_data = data.loc[sub_meta.index, :]
            sub_meta = sub_meta.reset_index().set_index('hash').loc[
                sub_data.index, :]
            # Buld target vector
            target = (sub_meta.loc[sub_data.index, target_col]).astype(int)

            logging.info('Class balance: %0.2f, Nr. of samples: %i' % ((target == 1).astype(
                float).mean(), len(target)))
            score = categorize(target, sub_data, target_time_point)
            score.loc[:, 'mc<0.5'] = mc
            scores.append(score)
        return pd.concat(scores)
    else:
        meta = meta.reset_index().set_index('hash').loc[data.index, :]
        # Buld target vector
        target = (meta.loc[data.index, target_col]).astype(int)
        logging.info('Class balance: %0.2f, Nr. of samples: %i' % ((target == 1).astype(
            float).mean(), len(target)))

        return categorize(target, data, target_time_point)


def confidence_decoder(meta, data, area, latency=0, signed=True):
    '''

    '''

    # Map to nearest time point in data
    times = np.unique(data.reset_index().time)
    target_time_point = times[np.argmin(abs(times - latency))]
    times_idx = np.isclose(data.time, target_time_point, 0.00001)
    data = data.loc[times_idx, :]

    # Turn data into (trial X Frequency)
    data = (data.reset_index()
                .loc[:, ['trial', 'est_val', area]]
                .set_index(['trial', 'est_val'])
                .unstack())

    meta = meta.reset_index().set_index('hash').loc[data.index, :]
    data = data.loc[meta.index, :]
    # Buld target vector
    if signed:
        target = (meta.loc[data.index, 'confidence'] *
                  meta.loc[data.index, 'response']).astype(int)
    else:
        target = (meta.loc[data.index, 'confidence'] == 1).astype(int)
    score = categorize(target, data, target_time_point)
    return score


def multiclass_roc(y_true, y_predict, **kwargs):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(label_binarize(y_true, classes=[-2, -1, 1, 2]),
                         label_binarize(y_predict, classes=[-2, -1, 1, 2]),
                         **kwargs)


def categorize(target, data, latency):
    '''
    Expects a pandas series and a pandas data frame.
    Both need to be indexed with the same index.
    '''
    from imblearn.pipeline import Pipeline
    from sklearn.metrics.scorer import make_scorer
    from sklearn.metrics import recall_score, precision_score
    from sklearn.utils.multiclass import type_of_target

    if not all(target.index.values == data.index.values):
        raise RuntimeError('Target and data not aligned with same index.')

    target = target.values
    data = data.values
    # Determine prediction target:
    y_type = type_of_target(target)
    if y_type == 'multiclass':
        metrics = {'roc_auc': make_scorer(multiclass_roc,
                                          average='weighted'),
                   'accuracy': 'accuracy'}
        #'recall': make_scorer(recall_score,
        #                      average='weighted'),
        #'precision': make_scorer(precision_score,
        #                         average='weighted')}
    else:
        metrics = ['roc_auc', 'accuracy', 'precision', 'recall']
    classifiers = {
        'SCVrbf': svm.SVC(kernel='rbf'),
        'SCVlin': svm.SVC(kernel='linear'),
        'LDA': discriminant_analysis.LinearDiscriminantAnalysis(
            shrinkage='auto', solver='eigen'),
        'RForest': RandomForestClassifier()}

    scores = []
    for name, clf in list(classifiers.items()):
        clf = Pipeline([
            ('Scaling', StandardScaler()),
            #('PCA', PCA(n_components=20)),
            ('Upsampler', RandomOverSampler(sampling_strategy='minority')),
            (name, clf)
        ])
        # print('Running', name)
        score = cross_validate(clf, data, target,
                               cv=10, scoring=metrics,
                               return_train_score=False,
                               n_jobs=n_jobs)
        del score['fit_time']
        del score['score_time']
        score = {k: np.mean(v) for k, v in list(score.items())}
        score['latency'] = latency
        score['Classifier'] = name
        scores.append(score)
    return pd.DataFrame(scores)


def SVMCV(params):
    return RandomizedSearchCV(svm.SVC(), params, n_iter=50, n_jobs=4)


@memory.cache(ignore=['data'])
def get_subject(subject, area, epoch='stimulus', time=(0, 1),
                est_vals=(30, 100), est_key='F',
                baseline_epoch='stimulus', baseline=(-0.35, 0),
                data=None):
    if data is None:
        data = pd.concat([
            get_session(subject, session, area, epoch=epoch, time=time,
                        est_vals=est_vals, est_key=est_key,
                        baseline_epoch=baseline_epoch, baseline=baseline)
            for session in [0, 1, 2, 3]])
    else:
        col_select = list(area) + ['est_val', 'est_key', 'time', 'trial']
        data = data.reindex(col_select, axis=1)
        Q = '(%f<=time) & (time<=%f) & (%f<=est_val) & (est_val<= %f) & (est_key=="%s")' % (
            time[0], time[1], est_vals[0], est_vals[1], est_key)
        data = data.query(Q)

    meta = preprocessing.get_meta_for_subject(subject, epoch)

    meta.set_index('hash', inplace=True)
    return data, meta.loc[np.unique(data.trial), :]


#@memory.cache
def get_session(subject, session, area, epoch='stimulus', time=(0, 1),
                est_vals=(30, 100), est_key='F',
                baseline_epoch='stimulus', baseline=(-0.35, 0)):
    df = get_all_areas(subject, session, epoch=epoch, time=time,
                       est_vals=est_vals, est_key=est_key,
                       baseline_epoch=baseline_epoch, baseline=baseline)
    if not any([a in df.columns for a in ensure_iter(area)]):
        raise RuntimeError(
            '''Requested area %s does not exist. These are the known areas:
%s
            ''' % (area, str([x for x in df.columns])))
    cols = ['trial', 'time', 'est_key', 'est_val'] + list(ensure_iter(area))
    df = df.loc[:, cols].reset_index()
    df.loc[:, 'subject'] = subject
    df.loc[:, 'session'] = session
    return df.loc[:, ~df.columns.duplicated()]


def get_tableau(meta, dresp, areas={'lh': 'M1-lh', 'rh': 'M1-rh'},
                field='response',
                options=[1, 1, -1, -1],
                dbase=None, late=True,
                ** kwargs):
    '''
    Computes the response lateralization 'Tableau', i.e. Hemishphere*response
    plot.s
    '''
    rois = list(areas.values()) + ['time', 'trial', 'est_key', 'est_val']
    import pylab as plt
    # dresp = dresp.reset_index().loc[:, rois].query('est_key=="F"')
    # dresp.set_index(['trial', 'time', 'est_key', 'est_val'], inplace=True)
    dresp = dresp.reset_index().set_index('trial')
    plt.clf()

    meta = meta.loc[np.unique(dresp.index.values), :]
    for i, (resp, area) in enumerate(zip(options,
                                         [areas['lh'], areas['rh'],
                                          areas['lh'], areas['rh']])):
        plt.subplot(2, 2, i + 1)

        index = meta.query('%s==%i' % (field, resp)).index
        data = dresp.loc[index, :]

        k = pd.pivot_table(data, values=area,
                           index='est_val', columns='time')

        dtmin, dtmax = k.columns.min(), k.columns.max()
        dfmin, dfmax = k.index.min(), k.index.max()
        plt.imshow(np.flipud(k), cmap='RdBu_r',
                   aspect='auto', extent=[dtmin, dtmax, dfmin, dfmax], **kwargs)
        plt.text(0, 20, '%s:%i, HEMI=%s' % (field, resp, area))
        if i == 0:
            plt.title('LH')
            plt.xlabel('%s=%i' % (field, resp))
        if i == 1:
            plt.title('RH')


def get_path(epoch, subject, session, cache=False):
    from os.path import join
    from glob import glob

    if not cache:
        path = metadata.sr_labeled
        if epoch == 'stimulus':
            filenames = glob(
                join(path, 'S%i-SESS%i-stimulus-F-chunk*-lcmv.hdf' % (
                    subject, session)))
        elif epoch == 'response':
            filenames = glob(
                join(path, 'S%i-SESS%i-response-F-chunk*-lcmv.hdf' % (
                    subject, session)))
        else:
            raise RuntimeError('Do not understand epoch %s' % epoch)
    else:
        path = metadata.sraggregates
        if epoch == 'stimulus':
            filenames = join(
                path,  'S%i-SESS%i-stimulus-lcmv.hdf' % (
                    subject, session))
        elif epoch == 'response':
            filenames = join(
                path, 'S%i-SESS%i-response-lcmv.hdf' % (
                    subject, session))
    return filenames


def aggregate_files(globstring, hemis):
    from glob import glob
    tfr_data = pd.concat([pd.read_hdf(f) for f in glob(globstring)])
    return new_aggregates(tfr_data, hemis)


def new_aggregates(tfr_data, hemis):
    """Aggregate individual areas into clusters.
    """
    from pymeg import atlas_glasser
    all_clusters, _, _, _ = atlas_glasser.get_clusters()
    cluster_contrasts = []
    for hemi, cluster in product(hemis, all_clusters.keys()):
        tfrs_rh = [area for area in all_clusters[cluster] if 'rh' in area]
        tfrs_lh = [area for area in all_clusters[cluster] if 'lh' in area]

        left = tfr_data.loc[:, tfrs_lh].groupby(
            ['freq', 'area', 'time', 'trial']).mean().mean(1)
        right = tfr_data.loc[:, tfrs_rh].groupby(
            ['freq', 'area', 'time', 'trial']).mean().mean(1)

        hemi_name = 'Lateralized'
        if hemi == 'lh_is_ipsi':
            tfrs = left - right
        elif hemi == 'rh_is_ipsi':
            tfrs = right - left
        else:
            hemi_name = 'Averaged'
            tfrs = (right + left) / 2
        tfrs.loc[:, 'cluster'] = cluster
        tfrs.loc[:, 'hemi'] = hemi_name
        cluster_contrasts.append(tfrs)
    return pd.concat(cluster_contrasts)


def get_all_areas(subject, session, epoch='stimulus', time=(-1.5, 1),
                  est_vals=(10, 150), est_key='F',
                  baseline_epoch='stimulus', baseline=(-0.35, 0)):
    df = get_aggregates(subject, session, epoch=epoch,
                        baseline_epoch=baseline_epoch, baseline=baseline)
    df.query('%f<time & time<%f & est_key=="%s" & %f<=est_val & est_val<=%f' %
             (time[0], time[1], est_key, est_vals[0], est_vals[1]),
             inplace=True)
    return df


def submit_aggregates(cluster='uke'):
    from pymeg import parallel
    for subject, epoch, session in product(range(1, 16),
                                           ['stimulus', 'response'],
                                           range(4)):
        parallel.pmap(get_aggregates, [(subject, session, epoch)],
                      name='agg' + str(session) + epoch + str(subject),
                      tasks=5, memory=40
                      )


def get_aggregates(subject, session, epoch='stimulus',
                   baseline_epoch='stimulus', baseline=(-0.35, 0)):
    est_key = 'F'
    cachefile = get_path(epoch, subject, session, cache=True)

    # try:
    return pd.read_hdf(cachefile, 'epochs')
    # except IOError:

    filenames = get_path(epoch, subject, session, cache=False)
    meta = preprocessing.get_meta_for_subject(subject, epoch)
    meta = meta.set_index('hash')
    df = pd.concat([pd.read_hdf(f) for f in filenames])
    df = np.log10(df) * 10

    # This is the place where baselining should be carried out (i.e. before
    # averaging across areas)
    if baseline is not None:
        if not baseline == epoch:
            filenames = get_path('stimulus', subject, session, cache=False)
            dbase = pd.concat([pd.read_hdf(f) for f in filenames])
            dbase = np.log10(dbase) * 10
        else:
            dbase = df
        dbase = dbase.query(
            '%f<time & time<%f & est_key=="%s"' %
            (baseline[0], baseline[1], est_key))

    df.query('est_key=="%s" ' %
             (est_key),
             inplace=True)

    if baseline is not None:
        logging.info('Doing baseline correction')
        df = apply_baseline(df, dbase, trial=False)
        del dbase
        df.set_index(
            ['trial', 'time', 'est_key', 'est_val'], inplace=True)
    df = rois.reduce(df).reset_index()  # Reduce to visual clusters
    # Filter down to correct trials:
    meta = meta.reindex(np.unique(df.trial.values))
    df.set_index('trial', inplace=True)

    # Now compute lateralisation
    def lateralize(response):
        '''
        Expects a DataFrame with responses of one hand
        '''
        left, right = sorted(rois.lh(df.columns)), sorted(rois.rh(df.columns))
        if len(left) < len(right):
            right = sorted(l.replace('lh', 'rh') for l in left)
        elif len(left) > len(right):
            left = sorted(l.replace('rh', 'lh') for l in right)
        return rois.lateralize(response, left, right, '_L-R')

    response_Mone = df.loc[meta.response == -1, :]
    response_Pone = df.loc[meta.response == +1, :]
    if subject <= 8:
        lateralized = pd.concat(
            [lateralize(response_Mone),
             lateralize(response_Pone)])
    else:
        lateralized = pd.concat(
            [lateralize(response_Mone),
             lateralize(response_Pone)])
    # Averge hemispheres
    left, right = sorted(rois.lh(df.columns)), sorted(rois.rh(df.columns))

    if len(left) > len(right):
        right = sorted(l.replace('lh', 'rh') for l in left)
    elif len(left) < len(right):
        left = sorted(l.replace('rh', 'lh') for l in right)
    havg = pd.concat(
        [df.loc[:, (x, y)].mean(1) for x, y in zip(left, right)],
        1)
    havg.columns = [x + '_Havg' for x in left]
    assert(len(havg) == len(lateralized) == len(df))
    df = pd.concat((df.reset_index(), lateralized.reset_index(),
                    havg.reset_index()), 1)
    df = df.loc[:, ~df.columns.duplicated()]
    df.to_hdf(cachefile, 'epochs')
    return df


def apply_baseline(data, baseline, trial=False):
    '''
    Baseline with mean and variance across time
    '''
    dmean = baseline.groupby(
        ['est_val', 'est_key', 'time']).mean()  # Avg across trials
    dstd = dmean.groupby(
        ['est_val', 'est_key']).std()  # Avg across time
    dmean = dmean.groupby(['est_val', 'est_key']).mean()
    if 'time' in dmean.columns:
        del dmean['time'], dstd['time']
    index_cols = set(data.reset_index().columns) - set(dmean.columns)
    data = data.reset_index().set_index(list(index_cols))
    # target_cols = data.columns
    # (est_key, est_val) x areas
    # del dmean['trial'], dstd['trial']
    # areas = dmean.columns
    acc = []
    for cond, data in data.groupby(['est_val', 'est_key']):
        data = data.copy()
        means = dmean.loc[cond, data.columns]
        stds = dstd.loc[cond, data.columns]
        data.loc[:, :] = (data.values - means.values[np.newaxis, :]
                          ) / stds.values[np.newaxis, :]
        acc.append(data)
    data = pd.concat(acc)
    # data = data.join(dmean, on=['est_val', 'est_key'], rsuffix='_!mbase')
    # data = data.join(dstd, on=['est_val', 'est_key'], rsuffix='_!sbase')

    # for area in areas:
    #    data.loc[:, area] = data.loc[:, area] - \
    #        data.loc[:, area + '_!mbase']
    #    data.loc[:, area] = data.loc[:, area] / \
    #        data.loc[:, area + '_!sbase']

    return data.reset_index()  # .loc[:, target_cols]


def ensure_iter(input):
    if isinstance(input, str):
        yield input
    else:
        try:
            for item in input:
                yield item
        except TypeError:
            yield input


def filter_latency(data, min, max):
    lat = data.index.get_level_values('latency').values
    return data.loc[(min < lat) & (lat < max), :]


def make_brain_plots(data, atype='Pairs', ssd_view=['cau']):
    # 1 AUC
    auc_limits = {'MIDC_split': (0.3, 0.7), 'MIDC_nosplit': (0.3, 0.7),
                  'CONF_signed': (0.3, 0.7), 'CONF_unsigned': (.3, 0.7),
                  'SIDE_nosplit': (0.3, 0.7)}
    '''
    df = data.test_roc_auc.Pairs.query(
        'epoch=="stimulus" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, 0.15, 0.2),
                         limits=auc_limits, epoch='stimulus', measure='auc')
    df = data.test_roc_auc.Pairs.query(
        'epoch=="response" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, -0.05, 0.05),
                         limits=auc_limits, epoch='response', measure='auc')

    df = data.test_roc_auc.loc[:, atype].query(
        'epoch=="response" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, 0.425, 0.475),
                         limits=auc_limits, epoch='response_late_' + 'atype', measure='auc')

    acc_limits = {'MIDC_split': (-0.2, 0.2), 'MIDC_nosplit': (-0.2, 0.2),
                  'CONF_signed': (-0.2, 0.2), 'CONF_unsigned': (-.2, 0.2),
                  'SIDE_nosplit': (-0.2, 0.2)}
    df = data.test_accuracy.Pairs.query(
        'epoch=="stimulus" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, -0.05, 0.05),
                         limits=acc_limits, epoch='stimulus', measure='accuracy')
    df = data.test_accuracy.Pairs.query(
        'epoch=="response" & ~(signal=="SSD") & Classifier=="SCVlin"')
    plot_summary_results(filter_latency(df, -0.05, 0.05),
                         limits=acc_limits, epoch='response', measure='accuracy')
    '''
    data_ssd = filter_latency(data.test_slope.Pairs.query(
        'epoch=="stimulus" & (signal=="SSD") & Classifier=="Ridge"'), 0.18, 0.19)
    for sample, sd in data_ssd.groupby('sample'):
        ssd_limits = {'SSD': (-0.08, 0.08)}
        plot_summary_results(sd, limits=ssd_limits,
                             epoch='stimulus',
                             measure='slope' + '_sample%i' % sample,
                             views=ssd_view)


def plot_summary_results(data, cmap='RdBu_r',
                         limits={'MIDC_split': (0.3, 0.7),
                                 'MIDC_nosplit': (0.3, 0.7),
                                 'CONF_signed': (0.3, 0.7),
                                 'CONF_unsigned': (.3, 0.7),
                                 'SIDE_nosplit': (0.3, 0.7),
                                 'SSD': (-0.05, 0.05)},
                         ex_sub='S04', measure='auc', epoch='response',
                         classifier='svc',
                         views=[['par', 'fro'], ['lat', 'med']]):
    from pymeg import roi_clusters as rois, source_reconstruction as sr
    from matplotlib import colors, cm

    # labels = sr.get_labels(ex_sub)
    labels = sr.get_labels(ex_sub)
    lc = rois.labels_to_clusters(labels, rois.all_clusters, hemi='lh')

    for signal, dsignal in data.groupby('signal'):
        vmin, vmax = limits[signal]
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        colortable = cm.get_cmap(cmap)
        cfunc = lambda x: colortable(norm(x))
        brain = plot_one_brain(dsignal, signal, lc, cfunc, ex_sub=ex_sub,
                               measure=measure, classifier=classifier,
                               epoch=epoch, views=views)
    return brain


#@memory.cache
def plot_one_brain(dsignal, signal, lc, cmap, ex_sub='S04', classifier='SCVlin',
                   epoch='response', measure='auc', views=[['par', 'fro'], ['lat', 'med']]):
    from surfer import Brain
    print('Creating Brain')
    brain = Brain(ex_sub, 'lh', 'inflated',  views=['lat'], background='w')
    # subjects_dir='/Users/nwilming/u/freesurfer_subjects/')
    print('Created Brain')
    ms = dsignal.mean()
    if (signal == 'CONF_signed') and (measure == 'accuracy'):
        plot_labels_on_brain(brain, lc, ms, cmap)
    if (signal == 'SSD'):
        plot_labels_on_brain(brain, lc, ms, cmap)
    else:
        plot_labels_on_brain(brain, lc, ms, cmap)
    brain.save_montage('/Users/nwilming/Desktop/%s_montage_%s_%s_%s.png' %
                       (signal, measure, classifier, epoch), views)
    return brain


def plot_labels_on_brain(brain, labels, data, cmap):
    already_plotted = []
    for label in data.index.values:
        for clustername, lobjects in labels.items():
            if clustername == label:
                value = data.loc[label]

                for l0 in lobjects:
                    if any(l0.name == x for x in already_plotted):
                        import pdb
                        pdb.set_trace()
                    # print(('Addding', l0.name, cmap(value)))
                    l0.color = cmap(value)
                    already_plotted.append(l0.name)
                    brain.add_label(l0, color=cmap(value), alpha=0.8)
    brain.save_image('test.png')


@memory.cache
def _dcd_helper_getter(path):
    import os
    df = pd.read_hdf(os.path.join(path, 'all_decoding_results.hdf'))
    df = df.reset_index().set_index(
        ['Classifier', 'epoch', 'est_key', 'latency', 'mc<0.5', 'signal', 'subject', 'area'])
    df = df.query('~(signal=="SSD")')
    df = df.unstack('area').T
    return df


@memory.cache
def _ssd_helper_getter(path):
    import os
    ssd = pd.read_hdf(os.path.join(path, 'all_ssd_results.hdf'))
    ssd.loc[:, 'latency'] = np.around(ssd.latency.astype(float), 4)
    ssd = ssd.reset_index().set_index(
        ['Classifier', 'epoch', 'est_key', 'latency', 'signal', 'subject', 'sample', 'area'])
    ssd = ssd.loc[~ssd.index.duplicated(), :]
    ssd = ssd.unstack('area').T

    return ssd


def recode(df):
    dt = []
    areas = []
    for area in df.index.get_level_values('area'):
        if '_L-R' in area:
            dt.append('Lateralized')
            areas.append(area.replace('-lh_L-R', ''))
        elif '_Havg' in area:
            dt.append('Average')
            areas.append(area.replace('-lh_Havg', ''))
        else:
            dt.append('Pairs')
            areas.append(area.split(',')[0].replace(
                '(', '').replace(')', '').replace("'", '').replace('-lh', ''))
    df.loc[:, 'atype'] = dt
    df.set_index('atype', append=True, inplace=True)
    df = df.swaplevel(1, 2)
    df = df.T
    df.columns.set_levels(areas, level='area', inplace=True)
    return df


def get_decoding_data(path='/home/nwilming/conf_meg/'):
    # files = glob.glob(os.path.join(path, 'concat_S*'))
    # df = pd.concat([pd.read_hdf(f) for f in files])
    df = recode(_dcd_helper_getter(path))
    ssd = recode(_ssd_helper_getter(path))

    idt = df.test_accuracy.index.get_level_values('signal') == 'CONF_signed'
    df.loc[idt, 'test_accuracy'] = (df.loc[idt, 'test_accuracy'] - 0.25).values
    df.loc[~idt, 'test_accuracy'] = (
        df.loc[~idt, 'test_accuracy'] - 0.5).values
    return df.query('~(subject==6)'), ssd


def plot_signals(data, measure, classifier='svm', ylim=(0.45, 0.75)):
    import pylab as plt
    for epoch, de in data.groupby('epoch'):
        print(de.shape)
        g = plot_by_signal(de)
        g.set_ylabels(r'$%s$' % measure)
        g.set_xlabels(r'$time$')
        g.set(ylim=ylim)
        plt.savefig('/Users/nwilming/Desktop/%s_%s_%s_decoding.pdf' %
                    (measure, classifier, epoch))


def plot_by_signal(data, signals={'MIDC_split': '#E9003A', 'MIDC_nosplit': '#FF5300',
                                  'CONF_signed': '#00AB6F', 'CONF_unsigned': '#58E000'}):
    import pylab as plt
    import seaborn as sns
    idsig = data.index.get_level_values('signal').isin(signals.keys())
    data = data.loc[idsig, :]
    split = data.index.get_level_values('mc<0.5').values.astype(float)

    nosplit = data.loc[np.isnan(split), :].groupby(
        ['latency', 'signal']).mean().stack().reset_index()

    concat_df = [nosplit]

    for split_ind, dsignal in data.loc[~np.isnan(split)].groupby('mc<0.5'):
        k = dsignal.groupby(['latency', 'signal']
                            ).mean().stack().reset_index()
        if not split_ind:
            k.loc[:, 0] *= -1
        k.columns = ['latency', 'signal', 'area', split_ind]
        concat_df.append(k)
    k = pd.concat(concat_df)

    g = sns.FacetGrid(k, col='signal', col_wrap=2, hue='area', palette='magma')
    g.map(plt.plot, 'latency', 0, alpha=0.8)
    g.map(plt.plot, 'latency', 1, alpha=0.8)

    #    k = data.groupby(['latency', 'signal']).mean().stack().reset_index()
    #    g = sns.FacetGrid(k, col='signal', col_wrap=2,
    #                      hue='area', palette='magma')
    #    g.map(plt.plot, 'latency', 0, alpha=0.8)
    return g


def plot_interesting_areas(data,
                           signals={'MIDC_split': '#E9003A', 'SIDE_nosplit': '#FF5300',
                                    'CONF_signed': '#00AB6F', 'CONF_unsigned': '#58E000'},
                           title='', classifier='SCVlin'):
    interesting_areas = ['visual', 'FEF', 'IPS_Pces', 'M1', 'aIPS1', 'Area6_dorsal_medial',
                         'Area6_anterior', 'A6si', 'PEF', '55b', '8av', '8C', '24dv']
    areas = []

    areas = [x for x in data.columns if any(
        [i in x for i in interesting_areas])]

    data = data.query('Classifier=="%s"' % classifier)
    plot_set(areas, data, title=title)


def plot_set(area_set, data,
             signals={'MIDC_split': '#E9003A', 'SIDE_nosplit': '#FF5300',
                      'CONF_signed': '#00AB6F', 'CONF_unsigned': '#58E000'},
             title=''):
    import matplotlib
    import seaborn as sns
    import pylab as plt
    gs = matplotlib.gridspec.GridSpec(1, 2)
    for i, area in enumerate(area_set):
        for signal, color in signals.items():
            try:
                plot_decoding_results(data, signal, area, stim_ax=gs[
                    0, 0], resp_ax=gs[0, 1], color=color,
                    offset=i * 0.5)
            except RuntimeError:
                print('RuntimeError for area %s, signal %s' % (area, signal))
    sns.despine(left=True)
    plt.suptitle(title)

    # Add legend
    x = [-0.5, -1, -1, -0.5]
    y = [0., 0.2, 0, 0.2]
    ylim = list(plt.ylim())
    ylim[1] = len(area_set) * 0.5 + 1
    for i, (signal, color) in enumerate(signals.items()):
        plt.text(x[i], y[i], signal, color=color)
    plt.subplot(gs[0, 0])
    plt.ylim(ylim[0] - 0.1, ylim[1])
    plt.subplot(gs[0, 1])
    plt.ylim(ylim[0] - 0.1, ylim[1])


def plot_decoding_results(data, signal, area,
                          stim_ax=None, resp_ax=None,  color='b',
                          offset=0):
    '''
    Data is a df that has areas as columns and at least subjct, classifier, latency and signal as index.
    Values of the dataframe encode the measure of choice to plot.
    '''
    import warnings
    warnings.filterwarnings("ignore")
    import pylab as plt
    import seaborn as sns
    if stim_ax is None:
        stim_ax = plt.gca()
    if resp_ax is None:
        stim_ax = plt.gca()
    data = data.loc[:, area]
    select_string = 'signal=="%s"' % (signal)
    areaname = (str(area).replace('vfc', '')
                .replace('-lh', '')
                .replace('-rh', '')
                .replace('_Havg', '')
                .replace('_Lateralized', ''))
    data = data.reset_index().query(select_string)
    if '_split' in signal:
        data = data.groupby(['subject', 'epoch', 'latency']
                            ).mean().reset_index()
    stimulus = data.query('epoch=="stimulus"').reset_index()
    stimulus.loc[:, area] += offset

    response = data.query('epoch=="response"').reset_index()
    response.loc[:, area] += offset
    stim_ax = plt.subplot(stim_ax)
    sns.tsplot(stimulus, time='latency', value=area,
               unit='subject', ax=stim_ax, color=color)
    # plt.ylim([0.1, 0.9])
    plt.axhline(0.5 + offset, color='k')
    dx, dy = np.array([0.0, 0.0]), np.array([.5, 0.75])
    plt.plot(dx, dy + offset, color='k')
    plt.text(-0.75, 0.6 + offset, areaname)
    plt.yticks([])
    plt.ylabel('')
    resp_ax = plt.subplot(resp_ax)

    sns.tsplot(response, time='latency', value=area,
               unit='subject', ax=resp_ax, color=color)
    plt.plot(dx, dy + offset, color='k')
    plt.yticks([])
    plt.ylabel('')
    # plt.ylim([0.1, 0.9])
    plt.axhline(0.5 + offset, color='k')


@memory.cache
def do_stats(x):
    from mne.stats import permutation_cluster_1samp_test
    return permutation_cluster_1samp_test(
        x, threshold=dict(start=0, step=0.2))


def plot_ssd_per_sample(ssd, cmap='magma', alpha=0.05, latency=0.18):
    import pylab as plt
    import seaborn as sns
    sns.set_style('ticks')
    cmap = plt.get_cmap(cmap)
    plt.figure(figsize=(6, 3.3))
    for sample, ds in ssd.test_corr.Average.groupby('sample'):
        k = ds.groupby(['subject', 'latency']).mean().reset_index()
        k = pd.pivot_table(ds.reset_index(), columns='latency',
                           index='subject', values='vfcvisual')
        baseline = k.loc[:, :0].mean(axis=1)
        baseline_corrected = k.sub(baseline, axis=0)

        t_obs, clusters, cluster_pv, H0 = do_stats(baseline_corrected.values)
        plt.plot(k.columns.values + 0.1 * sample,
                 k.values.mean(0), color=cmap(sample / 10.))
        sig_x = (k.columns.values + 0.1 * sample)[cluster_pv < alpha]
        plt.plot(sig_x, sig_x * 0 - 0.0001 *
                 np.mod(sample, 2), color=cmap(sample / 10))
        plt.plot([0.1 * sample + latency, 0.1 * sample + latency],
                 [0, 0.08], color='k', alpha=0.25, zorder=-1)
    plt.xlabel(r'$time$')
    plt.ylabel(r'$contrast \sim power$')
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(
        '/Users/nwilming/Dropbox/UKE/confidence_study/ssd_slopes_corr.svg')


def extract_peak_slope(ssd, latency=0.18):
    ulat = np.unique(ssd.index.get_level_values('latency'))
    latency = ulat[np.argmin(np.abs(ulat - latency))]
    print(latency)
    ssd = ssd.query('latency==%f & Classifier=="Ridge"' %
                    latency).test_slope.Average.loc[:, 'vfcvisual']
    return ssd.reset_index().loc[:, ('subject', 'sample', 'vfcvisual')]


def extract_kernels():
    pass
