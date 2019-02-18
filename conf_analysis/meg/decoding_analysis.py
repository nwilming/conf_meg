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

from os.path import join

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


from joblib import Memory

if 'TMPDIR' in os.environ.keys():
    memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'])
    inpath = '/nfs/nwilming/MEG/sr_labeled/aggs'
    outpath = '/nfs/nwilming/MEG/sr_decoding/'
elif 'RRZ_LOCAL_TMPDIR' in os.environ.keys():
    tmpdir = os.environ['RRZ_LOCAL_TMPDIR']
    outpath = '/work/faty014/MEG/sr_labeled/aggs/'
    outpath = '/work/faty014/MEG/sr_decoding/'
    memory = Memory(cachedir=tmpdir)
else:
    inpath = '/home/nwilming/conf_meg/sr_labeled/aggs'
    outpath = '/home/nwilming/conf_meg/sr_decoding'
    memory = Memory(cachedir=metadata.cachedir)

n_jobs = 10

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


def decoders():
    return {'SSD': ssd_decoder,
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
            'SIDE_nosplit': partial(midc_decoder, splitmc=False,
                                    target_col='side'),
            'CONF_signed': partial(midc_decoder, splitmc=False,
                                   target_col='signed_confidence'),
            'CONF_unsigned': partial(midc_decoder, splitmc=False,
                                     target_col='unsigned_confidence'),
            'CONF_unsign_split': partial(midc_decoder, splitmc=True,
                                         target_col='unsigned_confidence')}


def set_n_threads(n):
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['OMP_NUM_THREADS'] = str(n)


def submit(cluster='SLURM'):
    from pymeg import parallel
    decoder = decoders().keys()
    if cluster == 'SLURM':
        pmap = partial(parallel.pmap, email=None, tasks=1, nodes=1,
                       memory=60,
                       ssh_to=None,
                       walltime='72:00:00',  # walltime='72:00:00',
                       cluster='SLURM',
                       env='py36')
    else:
        pmap = partial(parallel.pmap, nodes=1, tasks=n_jobs, memory=61,
                       ssh_to=None,  walltime='72:00:00', env='py36')

    for subject, epoch, dcd in product([1, 16],
                                       ['stimulus', 'response'],
                                       decoder):
        pmap(run_decoder, [(subject, dcd, epoch)],
             name='DCD' + dcd + epoch + str(subject),
             )


def get_save_path(subject, decoder, area, epoch):
    filename = 'S%i-%s-%s-%s-decoding.hdf' % (
        subject, decoder, epoch, area)
    return join(outpath, filename)


def augment_meta(meta):
    meta.loc[:, 'signed_confidence'] = (meta.loc[:, 'confidence'] *
                                        meta.loc[:, 'response']).astype(int)
    meta.loc[:, 'unsigned_confidence'] = (
        meta.loc[:, 'confidence'] == 1).astype(int)
    return meta


def run_decoder(subject, decoder, epoch, ntasks=n_jobs,
                hemis=['Lateralized', 'Averaged', 'Pair']):
    '''
    Parallelize across areas and hemis.
    '''
    set_n_threads(1)
    from multiprocessing import Pool
    from glob import glob
    from pymeg import atlas_glasser, aggregate_sr as asr
    clusters, _, _, _ = atlas_glasser.get_clusters()
    areas = clusters.keys()
    filenames = glob(join(inpath, 'S%i_*_%s_agg.hdf' % (subject, epoch)))

    meta = augment_meta(
        preprocessing.get_meta_for_subject(subject, 'stimulus'))
    #meta = meta.dropna(subset=['contrast_probe'])
    args = []
    for area, hemi in product(areas, hemis):
        if hemi == 'Pair':
            area = [area + '_RH', area + '_LH']

        args.append((meta, asr.delayed_agg(filenames,
                                           hemi=hemi,
                                           cluster=area),
                     decoder, hemi))
    print('Processing %i tasks' % (len(args)))

    with Pool(ntasks) as p:
        scores = p.starmap(_par_apply_decode, args)  # , chunksize=ntasks)
    scores = [s for s in scores if s is not None]
    scores = pd.concat(scores)
    filename = outpath + '/concat_S%i-%s-%s-decoding.hdf' % (
        subject, decoder, epoch)
    scores.loc[:, 'subject'] = subject
    scores.to_hdf(filename, 'decoding')

    p.terminate()
    return scores


def _par_apply_decode(meta, delayed_agg, decoder, hemi=None):
    agg = delayed_agg()
    dt = np.diff(agg.columns.get_level_values('time'))[0]
    latencies = None
    if 'SSD' in decoder:
        latencies = np.arange(-.1, 0.75, dt)

    return apply_decoder(meta, agg, decoders()[decoder], latencies=latencies,
                         hemi=hemi)


def apply_decoder(meta, agg, decoder, latencies=None, hemi=None):
    """Run a decoder across a set of latencies.

    Args:
        agg: pd.DataFrame
            Aggregate data frame.
        decoder: function
            Function that takes meta data, agg data, areas
            and latency as input and returns decoding scores
            as data frame.
        latencies: None or list
            List of latencies to decode across. If None
            all time points in agg will be used.
    """
    import time
    start = time.time()
    if latencies is None:
        latencies = agg.columns.get_level_values('time').values

    print(agg, decoder)
    print(agg.head())
    
    area = np.unique(agg.index.get_level_values('cluster'))
    scores = []
    for latency in latencies:
        logging.info('Applying decoder %s at latency %s' % (decoder, latency))
        try:
            s = decoder(meta, agg, area, latency=latency)
            s.loc[:, 'cluster'] = str(area)
            scores.append(s)
        except:
            logging.exception(''''Error in run_decoder:        
        # Decoder: %s
        # Area: %s
        # Latency: %f)''' % (str(decoder), area, latency))
            raise
    res = pd.concat(scores)
    res.loc[:, 'hemi'] = hemi
    logging.info('Applying decoder %s across N=%i latencies took %3.2fs' % (
        decoder, len(latencies), time.time() - start))
    return res


def ssd_decoder(meta, data, area, latency=0.18, target_value='contrast'):
    '''
    Sensory signal decoder (SSD).

    Each frequency and brain area is one feature, contrast is the target.
    Args:
        meta: DataFrame
            Metadata frame that contains meta data per row
        data: Aggregate data frame
        area: List or str
            Which areas to use for decoding. Multiple areas provide
            independent features per observation
        latency: float
            Which time point to decode
        target_value: str
            Which target to decode (refers to a col. in meta)

    '''

    from sklearn.metrics import make_scorer
    from scipy.stats import linregress
    slope_scorer = make_scorer(lambda x, y: linregress(x, y)[0])
    corr_scorer = make_scorer(lambda x, y: np.corrcoef(x, y)[0, 1])
    meta = meta.set_index('hash')
    sample_scores = []

    for sample_num, sample in enumerate(np.arange(0, 1, 0.1)):
        # Map sample times to existing time points in data
        target_time_point = sample + latency
        times = data.columns.get_level_values('time').values
        target_time_point = times[np.argmin(abs(times - target_time_point))]

        # Compute mean latency
        latency = target_time_point - sample

        # Turn data into (trial X Frequency)
        X = []
        for a in ensure_iter(area):
            x = pd.pivot_table(data.query('cluster=="%s"' % a), index='trial',
                               columns='freq', values=target_time_point)
            X.append(x)
        sample_data = pd.concat(X, 1)

        # Build target vector
        cvals = np.stack(meta.loc[sample_data.index, 'contrast_probe'].values)
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
                                   n_jobs=1)  # n_jobs = 1 because it
            # is nested par loop
            # fit = clf(sample_data.values, target)
            del score['fit_time']
            del score['score_time']
            score = {k: np.mean(v) for k, v in list(score.items())}
            score['latency'] = latency
            score['Classifier'] = name
            score['sample'] = sample_num
            # score['coefs'] = fit.coef_.astype(object)
            sample_scores.append(score)
        del sample_data

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
    meta = meta.set_index('hash')
    # Map to nearest time point in data
    times = data.columns.get_level_values('time').values
    target_time_point = times[np.argmin(abs(times - latency))]
    logging.info('Selecting next available time point: %02.2f' %
                 target_time_point)
    data = data.loc[:, target_time_point]

    # Turn data into (trial X Frequency)
    X = []
    for a in ensure_iter(area):
        x = pd.pivot_table(data.query('cluster=="%s"' % a), index='trial',
                           columns='freq', values=target_time_point)
        X.append(x)
    data = pd.concat(X, 1)
    meta = meta.loc[data.index, :]
    scores = []
    if splitmc:
        for mc, sub_meta in meta.groupby(meta.mc < 0.5):
            sub_data = data.loc[sub_meta.index, :]
            sub_meta = sub_meta.loc[sub_data.index, :]
            # Buld target vector
            target = (sub_meta.loc[sub_data.index, target_col]).astype(int)
            logging.info('Class balance: %0.2f, Nr. of samples: %i' % ((target == 1).astype(
                float).mean(), len(target)))
            score = categorize(target, sub_data, target_time_point)
            score.loc[:, 'mc<0.5'] = mc
            scores.append(score)
        scores = pd.concat(scores)
    else:
        # Buld target vector
        target = (meta.loc[data.index, target_col]).astype(int)
        logging.info('Class balance: %0.2f, Nr. of samples: %i' % ((target == 1).astype(
            float).mean(), len(target)))

        scores = categorize(target, data, target_time_point)
    scores.loc[:, 'area'] = str(area)
    scores.loc[:, 'latency'] = latency
    return scores


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
                               n_jobs=1)
        del score['fit_time']
        del score['score_time']
        score = {k: np.mean(v) for k, v in list(score.items())}
        score['latency'] = latency
        score['Classifier'] = name
        scores.append(score)
    return pd.DataFrame(scores)


def SVMCV(params):
    return RandomizedSearchCV(svm.SVC(), params, n_iter=50, n_jobs=4)


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


def submit_aggregates(cluster='uke'):
    from pymeg import parallel
    for subject, epoch, session in product([7],
                                           ['stimulus'],
                                           range(0, 1)):
        parallel.pmap(aggregate, [(subject, session, epoch)],
                      name='agg' + str(session) + epoch + str(subject),
                      tasks=8, memory=60, walltime='12:00:00',
                      )


def aggregate(subject, session, epoch):
    from pymeg import aggregate_sr as asr
    from os.path import join
    stim = (
        '/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-stimulus-*-chunk*-lcmv.hdf' % (
            subject, session))
    resp = (
        '/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-response-*-chunk*-lcmv.hdf' % (
            subject, session))

    if epoch == 'stimulus':
        agg = asr.aggregate_files(stim, stim, (-0.25, 0))
    elif epoch == 'response':
        agg = asr.aggregate_files(resp, stim, (-0.25, 0))

    filename = join(
        '/home/nwilming/conf_meg/sr_labeled/aggs/',
        'S%i_SESS%i_%s_agg.hdf' % (subject, session, epoch))
    asr.agg2hdf(agg, filename)


def ensure_iter(input):
    if isinstance(input, str):
        yield input
    else:
        try:
            for item in input:
                yield item
        except TypeError:
            yield input
