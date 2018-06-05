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

else:
    memory = Memory(cachedir=metadata.cachedir)

areas = [
    # Frontal areas:
    'ACC', 'f_inf_orbital', 'Sf_sup', 'f_inf_Triangul',
    'frontopol', 'Sf_middle', 'Gf_middle',  #'frontomargin',
    'Gf_sup', 'Sf_inf', 'f_inf_opercular',
    # Visual Clusters:
    'vfcIPS_occ', 'vfcTO', 'vfcLO', 'vfcIPS_dorsal', 'vfcSPL',
    'vfcPHC', 'vfcVO', 'vfcvisual', 'vfcFEF', 'vfcV3ab',
    # Choice areas:
    'aIPS1', 'M1', 'IPS_Pces'
]

avgs = [x + '-lh_Havg' for x in areas]
lateralized = [x + '-lh_L-R' for x in areas]
pairs = [(x + '-lh', x + '-rh') for x in areas]

areas = avgs + lateralized + pairs
n_jobs = 1



def submit():
    decoders = ['MIDC_split', 'MIDC_nosplit',
               'SIDE_split', 'SIDE_nosplit',
               'CONF_signed', 'CONF_unsigned']
    from pymeg import parallel
    for subject in range(1, 16):
        parallel.pmap(run_AA, [(subject, 'SSD', 'stimulus')], 
            home='/work/faty014', walltime='11:50:55', memory=60, nodes=1, tasks=1, 
            name='DCDSSDStimulus'+str(subject), ssh_to=None, cluster='SLURM', 
            email=None)
    for subject in range(1, 16):
         for epoch in ['stimulus', 'response']:
             for decoder in decoders:
                  parallel.pmap(run_AA, [(subject, decoder, epoch)], 
                      home='/work/faty014', walltime='11:50:55', memory=60, nodes=1, tasks=1, 
                      name='DCD'+decoder+epoch+str(subject), ssh_to=None, cluster='SLURM', 
                      email=None)


def run_AA(subject, decoder, epoch):
    from multiprocessing import Pool
    p = Pool(16)
    latencies = {'stimulus': [-0.5, 1.35], 'response': [-1, 0.5]}
    time = latencies[epoch]
    data = pd.concat([
            get_all_areas(subject, session, epoch=epoch, time=time,
	        est_vals=[10, 100], est_key='F', log10=True,
	        baseline_epoch='stimulus', baseline=(-0.35, 0))
            for session in [0, 1, 2, 3]])

    print('Done caching')
    from itertools import starmap
    scores = p.starmap(run_single_area, 
                       [(subject, decoder, epoch, area) for area in areas])
    scores = [s for s in scores if s is not None]
    scores = pd.concat(scores)
    filename = '/work/faty014/MEG/sr_decoding/S%i-%s-%s-decoding.hdf' % (
        subject, decoder, epoch)
    scores.to_hdf(filename, 'decoding')
    p.terminate()
    return scores


def run_single_area(subject, decoder, epoch, area, ignore_errors=True):
    try: 
        return run_decoder(subject, decoder, area, epoch)
    except:
        print('Error in runAA: (%i, %s, %s, %s)' %
                              (subject, decoder, epoch, area))
        if not ignore_errors:
            raise


def run_decoder(subject, decoder, area, epoch):
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
        if ('RRZ_TMPDIR' in list(os.environ.keys())) or ('RRZ_LOCAL_TMPDIR' in list(os.environ.keys())):
            path = '/work/faty014/MEG/'
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
                             est_vals=[10, 100])
    data.reset_index().set_index('trial', inplace=True)
    meta = meta.dropna(subset=['confidence', 'response'])
    
    dt = np.diff(data.time)[0]
    latencies = np.unique(data.time)
    meta.loc[:, 'signed_confidence'] = (meta.loc[:, 'confidence'] *
                                       meta.loc[:, 'response']).astype(int)
    meta.loc[:, 'unsigned_confidence'] = (
        meta.loc[:, 'confidence'] == 1).astype(int)

    decoders = {'SSD': ssd_decoder,
                'MIDC_split': partial(midc_decoder, splitmc=False,
                                      target_col='response'),
                'MIDC_nosplit': partial(midc_decoder, splitmc=False,
                                        target_col='response'),
		'SIDE_split': partial(midc_decoder, splitmc=False,
                                      target_col='side'),
                'SIDE_nosplit': partial(midc_decoder, splitmc=False,
                                        target_col='side'),
                'CONF_signed': partial(midc_decoder, splitmc=False,
                                       target_col='signed_confidence'),
                'CONF_unsigned': partial(midc_decoder, splitmc=False,
                                         target_col='unsigned_confidence')}

    if decoder == 'SSD':
        latencies = np.arange(-.1, 0.5, dt)

    assert(decoder in list(decoders.keys()))
    scores = []
    for latency in latencies:
        logging.info('S=%i; DCD=%s, EPOCH=%s; t=%f'%(subject, decoder, epoch, latency))
        try:
            s = decoders[decoder](meta, data, area, latency=latency)
            scores.append(s)
        except:
            logging.exception(''''Error in run_decoder:
        #Subject: %i
        #Decoder: %s
        #Epoch: %s
        #Area: %s
        #Latency: %f)''' % (subject, decoder, epoch, area, latency))
            raise
    scores = pd.concat(scores)
    scores.loc[:, 'signal'] = decoder
    scores.loc[:, 'subject'] = subject
    scores.loc[:, 'epoch'] = epoch
    scores.loc[:, 'area'] = str(area)
    scores.to_hdf(filename, 'decoding')
    return scores


def ssd_decoder(meta, data, area, latency=0.18):
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
    target_time_points = np.arange(0, 1, 0.1) + latency
    times = np.unique(data.reset_index().time)
    target_time_points = np.array(
        [times[np.argmin(abs(times - t))] for t in target_time_points]
    )
    # Compute mean latency
    latency = np.mean(target_time_points - np.arange(0, 1, 0.1))
    # print('Latency:', latency, '->', target_time_points)
    # Turn data into (trial X Frequency)
    X = []
    for a in ensure_iter(area):
        x = (data.reset_index()
             .loc[:, ['trial', 'time', 'est_val', a]]
             .set_index(['trial', 'time', 'est_val'])
             .unstack())
        X.append(x)
    data = pd.concat(X, 1)
   
    # Turn data into (trial*time X Frequency)
    #data = (data.reset_index()
    #            .loc[:, ['trial', 'time', 'est_val', area]]
    #            .set_index(['trial', 'time', 'est_val'])
    #            .unstack())
    # data.sort_index(inplace=True)
    data = data.loc[(slice(None), target_time_points), :]

    # Buld target vector
    target = np.concatenate(meta.contrast_probe.values)

    metrics = ['explained_variance', 'r2', 'neg_mean_squared_error']
    # from sklearn.kernel_ridge import KernelRidge
    # from sklearn.gaussian_process import GaussianProcessRegressor
    classifiers = {
        #'SVR': svm.SVR(),
        #'KRR': KernelRidge(kernel='rbf'), # Takes too long (>5 mins)
        #'GPR': GaussianProcessRegressor(), # Takes too long
        #'SGD': linear_model.SGDRegressor(max_iter=10),
        'OLS': linear_model.LinearRegression(),
        'Ridge': linear_model.RidgeCV()}
    #'Lasso': linear_model.LassoCV()}

    scores = []
    import time
    for name, clf in list(classifiers.items()):
        start = time.time()
        clf = Pipeline([
            ('Scaling', StandardScaler()),
            (name, clf)
        ])
        # print('Running', name, 'Latency', latency, end='')
        score = cross_validate(clf, data.values, target,
                               cv=10, scoring=metrics,
                               return_train_score=False,
                               n_jobs=n_jobs)
        del score['fit_time']
        del score['score_time']
        score = {k: np.mean(v) for k, v in list(score.items())}
        score['latency'] = latency
        score['Classifier'] = name
        scores.append(score)
        # print(' took: %3.2fs' % (time.time() - start))
    return pd.DataFrame(scores)


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
    logging.info('Selecting next available time point: %02.2f'%target_time_point)
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
            #sub_meta = meta
            #sub_data = data

            sub_data = data.loc[sub_meta.index, :]
            sub_meta = sub_meta.reset_index().set_index('hash').loc[
                sub_data.index, :]
            # Buld target vector
            target = (sub_meta.loc[sub_data.index, target_col]).astype(int)

            logging.info('Class balance: %0.2f, Nr. of samples: %i'%((target == 1).astype(
                float).mean(), len(target)))
            score = categorize(target, sub_data, target_time_point)
            score.loc[:, 'mc<0.5'] = mc
            scores.append(score)
        return pd.concat(scores)
    else:
        meta = meta.reset_index().set_index('hash').loc[data.index, :]
        # Buld target vector
        target = (meta.loc[data.index, target_col]).astype(int)
        logging.info('Class balance: %0.2f, Nr. of samples: %i'%((target == 1).astype(
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
    from imblearn.pipeline import Pipeline
    from sklearn.metrics.scorer import make_scorer
    from sklearn.metrics import recall_score, precision_score
    from sklearn.utils.multiclass import type_of_target

    # Determine prediction target:
    y_type = type_of_target(target)
    if y_type == 'multiclass':
        metrics = {'roc_auc': make_scorer(multiclass_roc,
                                          average='weighted'),
                   'accuracy': 'accuracy',
                   'recall': make_scorer(recall_score,
                                         average='weighted'),
                   'precision': make_scorer(precision_score,
                                            average='weighted')}
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
        score = cross_validate(clf, data.values, target,
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


def get_subject(subject, area, epoch='stimulus', time=(0, 1),
                est_vals=(30, 100), est_key='F', log10=True,
                baseline_epoch='stimulus', baseline=(-0.35, 0)):
    meta = preprocessing.get_meta_for_subject(subject, epoch)
    data = pd.concat([
        get_session(subject, session, area, epoch=epoch, time=time,
                    est_vals=est_vals, est_key=est_key, log10=log10,
                    baseline_epoch=baseline_epoch, baseline=baseline)
        for session in [0, 1, 2, 3]])
    meta.set_index('hash', inplace=True)
    return data, meta.loc[np.unique(data.trial), :]


#@memory.cache
def get_session(subject, session, area, epoch='stimulus', time=(0, 1),
                est_vals=(30, 100), est_key='F', log10=True,
                baseline_epoch='stimulus', baseline=(-0.35, 0)):
    df = get_all_areas(subject, session, epoch=epoch, time=time,
                       est_vals=est_vals, est_key=est_key, log10=log10,
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


def get_tableau(meta, dresp, dstim, baseline=None, log10=True,
                areas={'lh': 'M1-lh', 'rh': 'M1-rh'},
                **kwargs):
    '''
    Computes the response lateralization 'Tableau', i.e. Hemishphere*response
    plot.s
    '''
    rois = list(areas.values()) + ['time', 'trial', 'est_key', 'est_val']
    import pylab as plt
    dresp = dresp.reset_index().loc[:, rois].query('est_key=="F"')
    #dstim = dstim.reset_index().loc[:, rois].query('est_key=="F"')
    dresp.set_index(['trial', 'time', 'est_key', 'est_val'], inplace=True)
    #dstim.set_index(['trial', 'time', 'est_key', 'est_val'], inplace=True)
    if log10:
        dstim = np.log10(dstim) * 10
        dresp = np.log10(dresp) * 10
    dresp = dresp.reset_index().set_index('trial')
    #dstim = dstim.reset_index().set_index('trial')

    # Do baseline correction
    if baseline == 'avg':
        dstim = dstim.query('-0.3<time<-0')
        m = dstim.groupby('est_val').mean()
        s = (dstim.groupby(['est_val', 'trial'])
                  .mean()
                  .reset_index()
                  .groupby('est_val')
                  .std())

    for i, (resp, area) in enumerate(zip([-1, -1, 1, 1],
                                         [areas['lh'], areas['rh'],
                                          areas['lh'], areas['rh']])):
        plt.subplot(2, 2, i + 1)

        data = dresp.loc[meta.loc[meta.response == resp, :].index, :]
        k = pd.pivot_table(data, values=area,
                           index='est_val', columns='time')
        if baseline == 'avg':
            k = k.subtract(m.loc[:, area], 0).div(s.loc[:, area], 0)
        dtmin, dtmax = k.columns.min(), k.columns.max()
        plt.imshow(np.flipud(k), cmap='RdBu_r',
                   aspect='auto', extent=[dtmin, dtmax, 10, 100], **kwargs)
        plt.text(-1, 20, 'RESP:%i, HEMI=%s' % (resp, area))
        if i == 0:
            plt.title('LH')
            plt.xlabel('RESP=%i' % resp)
        if i == 1:
            plt.title('RH')


def get_path(epoch, subject, session):
    import os
    if 'RRZ_LOCAL_TMPDIR' in list(os.environ.keys()):
        path = '/work/faty014/MEG/'
    else:
        path = '/home/nwilming/conf_meg/'
    if epoch == 'stimulus':
        filename = 'sr_labeled/S%i-SESS%i-stimulus-lcmv.hdf' % (
            subject, session)
    elif epoch == 'response':
        filename = 'sr_labeled/S%i-SESS%i-response-lcmv.hdf' % (
            subject, session)
    else:
        raise RuntimeError('Do not understand epoch %s' % epoch)
    return os.path.join(path, filename)


@memory.cache
def get_all_areas(subject, session, epoch='stimulus', time=(0, 1),
                  est_vals=(30, 100), est_key='F', log10=True,
                  baseline_epoch='stimulus', baseline=(-0.35, 0)):
    print('.')
    meta = preprocessing.get_meta_for_subject(subject, epoch)
    meta = meta.set_index('hash')
    filename = get_path(epoch, subject, session)
    df = pd.read_hdf(filename)
    df.set_index(['trial', 'time', 'est_key', 'est_val'], inplace=True)
    if log10:
        df = np.log10(df) * 10

    # This is the place where baselining should be carried out (i.e. before
    # averaging across areas)
    if baseline is not None:
        if not baseline == epoch:
            filename = get_path('stimulus', subject, session)
            dbase = pd.read_hdf(filename)
            dbase.set_index(
                ['trial', 'time', 'est_key', 'est_val'], inplace=True)
            if log10:
                dbase = np.log10(dbase) * 10
        else:
            dbase = df
        dbase = dbase.query(
            '%f<time & time<%f & est_key=="%s" & %f<est_val & est_val<%f' %
            (baseline[0], baseline[1], est_key, est_vals[0], est_vals[1]))

    df.query('%f<time & time<%f & est_key=="%s" & %f<est_val & est_val<%f' %
             (time[0], time[1], est_key, est_vals[0], est_vals[1]),
             inplace=True)

    if baseline is not None:
        df = apply_baseline(df, dbase, trial=False)
        del dbase
        df.set_index(
            ['trial', 'time', 'est_key', 'est_val'], inplace=True)
    df = rois.reduce(df).reset_index()  # Reduce to visual clusters
    # Filter down to correct trials:
    meta = meta.loc[np.unique(df.trial.values), :]
    df.set_index('trial', inplace=True)

    left, right = rois.lh(df.columns), rois.rh(df.columns)

    # Now compute lateralisation
    def lateralize(response, response_hand):
        '''
        Expects a DataFrame with responses of one hand
        '''
        return rois.lateralize(response, left, right, '_L-R')

    response_Mone = df.loc[meta.response == -1, :]
    response_Pone = df.loc[meta.response == +1, :]
    if subject <= 8:
        lateralized = pd.concat(
            [lateralize(response_Mone, 'left'),
             lateralize(response_Pone, 'right')])
    else:
        lateralized = pd.concat(
            [lateralize(response_Mone, 'right'),
             lateralize(response_Pone, 'left')])
    # Averge hemispheres
    havg = pd.concat(
        [df.loc[:, (x, y)].mean(1) for x, y in zip(left, right)],
        1)
    havg.columns = [x + '_Havg' for x in left]
    return pd.concat(
        (df.reset_index(),
         lateralized.reset_index(),
         havg.reset_index()), 1)


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
    data = data.reset_index()
    target_cols = data.columns
    # (est_key, est_val) x areas
    #del dmean['trial'], dstd['trial']
    areas = dmean.columns
    data = data.join(dmean, on=['est_val', 'est_key'], rsuffix='_!mbase')
    data = data.join(dstd, on=['est_val', 'est_key'], rsuffix='_!sbase')

    for area in areas:
        data.loc[:, area] = data.loc[:, area] - \
            data.loc[:, area + '_!mbase']
        data.loc[:, area] = data.loc[:, area] / \
            data.loc[:, area + '_!sbase']

    return data.loc[:, target_cols]


def ensure_iter(input):
    if isinstance(input, str):
        yield input
    else:
        try:
            for item in input:
                yield item
        except TypeError:
            yield input
