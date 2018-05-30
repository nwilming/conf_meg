#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Decoding analyses for conf_meg data.

4.  Signed confidence and choice decoder: Same as MIDC and MDDC but with
    confidence folded into the responses (-2, -1, 1, 2)
5.  Unsigned confidence decoder: Same as MIDC and MDDC but decode
    confidence only.
'''
from __future__ import print_function
import logging
import numpy as np
import pandas as pd
import scipy as sp

from functools import partial

from sklearn import linear_model, discriminant_analysis, svm
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FactorAnalysis


from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler


from conf_analysis.behavior import metadata
from conf_analysis.meg import rois, preprocessing

from joblib import Memory

memory = Memory(cachedir=metadata.cachedir)

areas = [u'ACC-lh_Lateralized', u'vfcTO-lh_Lateralized',
         u'f_inf_orbital-lh_Lateralized', u'Sf_sup-lh_Lateralized',
         u'vfcLO-lh_Lateralized', u'vfcIPS_dorsal-lh_Lateralized',
         u'f_inf_Triangul-lh_Lateralized', u'frontopol-lh_Lateralized',
         u'Sf_middle-lh_Lateralized', u'IPS_Pces-lh_Lateralized',
         u'vfcSPL-lh_Lateralized', u'vfcPHC-lh_Lateralized',
         u'vfcVO-lh_Lateralized', u'Gf_middle-lh_Lateralized',
         u'vfcvisual-lh_Lateralized', u'vfcFEF-lh_Lateralized',
         u'M1-lh_Lateralized', u'frontomargin-lh_Lateralized',
         u'vfcV3ab-lh_Lateralized', u'Gf_sup-lh_Lateralized',
         u'Sf_inf-lh_Lateralized', u'vfcIPS_occ-lh_Lateralized',
         u'f_inf_opercular-lh_Lateralized', u'aIPS1-lh_Lateralized',
         u'ACC-lh_Havg', u'vfcTO-lh_Havg', u'f_inf_orbital-lh_Havg',
         u'Sf_sup-lh_Havg', u'vfcLO-lh_Havg', u'vfcIPS_dorsal-lh_Havg',
         u'f_inf_Triangul-lh_Havg', u'frontopol-lh_Havg', u'Sf_middle-lh_Havg',
         u'IPS_Pces-lh_Havg', u'vfcSPL-lh_Havg', u'vfcPHC-lh_Havg',
         u'vfcVO-lh_Havg', u'Gf_middle-lh_Havg', u'vfcvisual-lh_Havg',
         u'vfcFEF-lh_Havg', u'M1-lh_Havg', u'frontomargin-lh_Havg',
         u'vfcV3ab-lh_Havg', u'Gf_sup-lh_Havg', u'Sf_inf-lh_Havg',
         u'vfcIPS_occ-lh_Havg', u'f_inf_opercular-lh_Havg', u'aIPS1-lh_Havg']


n_jobs = 1


def submit():
    from pymeg import parallel
    for subject in range(1, 16):
        # parallel.pmap(run_AA, [(subject, 'SSD', 'stimulus')],
        #              walltime=10, memory=15, nodes='1:ppn=1',
        #              name='DCDSSDSTIM', ssh_to=None)
        for epoch in ['stimulus', 'response']:
            for decoder in ['MIDC']:
                parallel.pmap(
                    run_AA, [(subject, decoder, epoch)],
                    walltime=10, memory=15, nodes='1:ppn=1',
                    name='DCD' + decoder + epoch,
                    ssh_to=None)


def run_AA(subject, decoder, epoch):
    scores = []
    for area in areas:
        try:
            scores.append(run_decoder(subject, decoder, area, epoch))
        except:
            logging.exception('Error in runAA: (%i, %s, %s, %s)' %
                              (subject, decoder, epoch, area))
    scores = pd.concat(scores)
    filename = '/home/nwilming/conf_meg/sr_decoding/S%i-%s-%s-decoding.hdf' % (
        subject, decoder, epoch)
    scores.to_hdf(filename, 'decoding')
    return scores


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
    filename = '/home/nwilming/conf_meg/sr_decoding/S%i-%s-%s-%s-decoding.hdf' % (
        subject, decoder, epoch, area)
    try:
        k = pd.read_hdf(filename)
        logging.info('Loaded cached file: %s' % filename)
        return k
    except IOError:
        pass

    if epoch == 'stimulus':
        data, meta = get_subject(subject, epoch='stimulus',
                                 area=area, time=(-0.5, 1.35),
                                 est_vals=[10, 100])
        dt = np.diff(data.time)[0]
        latencies = np.arange(-.5, 1.35, dt)
    else:
        data, meta = get_subject(subject, epoch='response',
                                 area=area, time=(-1, 0.5),
                                 est_vals=[10, 100])
        dt = np.diff(data.time)[0]
        latencies = np.arange(-1, 0.5, dt)

    decoders = {'SSD': ssd_decoder,
                'MIDC': midc_decoder,
                'SCONF': partial(confidence_decoder, signed=True),
                'UCONF': partial(confidence_decoder, signed=True)}

    if decoder == 'SSD':
        latencies = np.arange(-.1, 0.5, dt)

    assert(decoder in decoders.keys())
    scores = []
    for latency in latencies:
        try:
            s = decoders[decoder](meta, data, area, latency=latency)
            scores.append(s)
        except:
            import pdb
            pdb.set_trace()
            logging.exception(''''Error in run_decoder:
Subject: %i
Decoder: %s
Epoch: %s
Area: %s
Latency: %f)''' % (subject, decoder, epoch, area, latency))

    scores = pd.concat(scores)
    scores.loc[:, 'signal'] = decoder
    scores.loc[:, 'subject'] = subject
    scores.loc[:, 'epoch'] = epoch
    scores.loc[:, 'area'] = area
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
    # Turn data into (trial*time X Frequency)
    data = (data.reset_index()
                .loc[:, ['trial', 'time', 'est_val', area]]
                .set_index(['trial', 'time', 'est_val'])
                .unstack())
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
        #'OLS': linear_model.LinearRegression(),
        'Ridge': linear_model.RidgeCV()}
    #'Lasso': linear_model.LassoCV()}

    scores = []
    import time
    for name, clf in classifiers.items():
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
        score = {k: np.mean(v) for k, v in score.items()}
        score['latency'] = latency
        score['Classifier'] = name
        scores.append(score)
        # print(' took: %3.2fs' % (time.time() - start))
    return pd.DataFrame(scores)


def midc_decoder(meta, data, area, latency=0):
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
    print('Selecting next available time point: ', target_time_point)
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
    # for mc, sub_meta in meta.groupby(meta.mc < 0.5):
    # Align data and meta
    sub_meta = meta
    sub_data = data
    sub_meta = sub_meta.reset_index().set_index('hash').loc[data.index, :]
    sub_data = data.loc[sub_meta.index, :]
    # Buld target vector
    target = (sub_meta.loc[sub_data.index, 'response']).astype(int)
    print(target.values)
    print('Class balance:', (target == 1).astype(float).mean())
    score = categorize(target, sub_data, target_time_point)
    # score.loc[:, 'mc<0.5'] = mc
    scores.append(score)
    return pd.concat(scores)


def confidence_decoder(meta, data, area, latency=0, signed=True):
    '''

    '''

    # Map to nearest time point in data
    times = np.unique(data.reset_index().time)
    target_time_point = times[np.argmin(abs(times - latency))]
    # print('Selecting next available time point: ', target_time_point)
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
    # print('Class balance:', target.astype(float).mean())
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
    for name, clf in classifiers.items():
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
        score = {k: np.mean(v) for k, v in score.items()}
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


@memory.cache
def get_session(subject, session, area, epoch='stimulus', time=(0, 1),
                est_vals=(30, 100), est_key='F', log10=True,
                baseline_epoch='stimulus', baseline=(-0.35, 0)):

    df = get_all_areas(subject, session, epoch=epoch, time=time,
                       est_vals=est_vals, est_key=est_key, log10=log10,
                       baseline_epoch=baseline_epoch, baseline=baseline)

    if not any([a in df.columns for a in area]):
        raise RuntimeError(
            '''Requested area does not exist. These are the known areas:
%s
            ''' % (str(df.columns)))
    cols = ['trial', 'time', 'est_key', 'est_val'] + list(ensure_iter(area))
    df = df.loc[:, cols].reset_index()
    df.loc[:, 'subject'] = subject
    df.loc[:, 'session'] = session
    return df.loc[:, ~df.columns.duplicated()]


def get_tableau(meta, dresp, dstim, baseline=None, log10=True,
                areas={'lh': 'lh.JWDG.lr_M1-lh', 'rh': 'rh.JWDG.lr_M1-rh'},
                **kwargs):
    rois = areas.values() + ['time', 'trial', 'est_key', 'est_val']
    import pylab as plt
    dresp = dresp.reset_index().loc[:, rois].query('est_key=="F"')
    dstim = dstim.reset_index().loc[:, rois].query('est_key=="F"')
    dresp.set_index(['trial', 'time', 'est_key', 'est_val'], inplace=True)
    dstim.set_index(['trial', 'time', 'est_key', 'est_val'], inplace=True)
    if log10:
        dstim = np.log10(dstim) * 10
        dresp = np.log10(dresp) * 10
    dresp = dresp.reset_index().set_index('trial')
    dstim = dstim.reset_index().set_index('trial')

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

        if baseline == 'avg':
            k = pd.pivot_table(data, values=area,
                               index='est_val', columns='time')
            k = k.subtract(m.loc[:, area], 0).div(s.loc[:, area], 0)

        plt.imshow(np.flipud(k), cmap='RdBu_r',
                   aspect='auto', extent=[-1, 0.5, 10, 100], **kwargs)
        plt.text(-1, 20, 'RESP:%i, HEMI=%s' % (resp, area))
        if i == 0:
            plt.title('LH')
            plt.xlabel('RESP=%i' % resp)
        if i == 1:
            plt.title('RH')


@memory.cache
def get_all_areas(subject, session, epoch='stimulus', time=(0, 1),
                  est_vals=(30, 100), est_key='F', log10=True,
                  baseline_epoch='stimulus', baseline=(-0.35, 0)):
    meta = preprocessing.get_meta_for_subject(subject, epoch)
    meta = meta.set_index('hash')
    if epoch == 'stimulus':
        filename = '/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-stimulus-lcmv.hdf' % (
            subject, session)
    elif epoch == 'response':
        filename = '/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-response-lcmv.hdf' % (
            subject, session)
    else:
        raise RuntimeError('Do not understand epoch %s' % epoch)
    df = pd.read_hdf(filename)
    df.set_index(['trial', 'time', 'est_key', 'est_val'], inplace=True)
    if log10:
        df = np.log10(df) * 10

    # This is the place where baselining should be carried out (i.e. before
    # averaging across areas)
    if baseline_epoch == epoch:
        dbase = df.query('%f < time %f' % (baseline))
        df.query('%f<time & time<%f & est_key=="%s" & %f<est_val & est_val<%f' %
                 (time[0], time[1], est_key, est_vals[0], est_vals[1]),
                 inplace=True)

    else:
        df.query('%f<time & time<%f & est_key=="%s" & %f<est_val & est_val<%f' %
                 (time[0], time[1], est_key, est_vals[0], est_vals[1]),
                 inplace=True)
        filename = '/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-stimulus-lcmv.hdf' % (
            subject, session)

        dbase = pd.read_hdf(filename)
        dbase.set_index(['trial', 'time', 'est_key', 'est_val'], inplace=True)
        if log10:
            dbase = np.log10(dbase) * 10
        dbase = dbase.query('%f < time < %f' % (baseline))
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
        ['est_val', 'est_key', 'trial']).mean()  # Avg across time
    dstd = baseline.groupby(
        ['est_val', 'est_key', 'trial']).std()  # Avg across time
    if 'time' in dmean.columns:
        del dmean['time'], dstd['time']
    data = data.reset_index()
    target_cols = data.columns
    if not trial:
        # (est_key, est_val) x areas
        dmean = dmean.groupby(['est_val', 'est_key']).mean()
        dstd = dstd.groupby(['est_val', 'est_key']).mean()
        #del dmean['trial'], dstd['trial']
        areas = dmean.columns
        data = data.join(dmean, on=['est_val', 'est_key'], rsuffix='_!mbase')
        data = data.join(dstd, on=['est_val', 'est_key'], rsuffix='_!sbase')

        for area in areas:
            data.loc[:, area] = data.loc[:, area] - \
                data.loc[:, area + '_!mbase']
            data.loc[:, area] = data.loc[:, area] / \
                data.loc[:, area + '_!sbase']
    else:
        areas = dmean.columns
        data = data.join(
            dmean, on=['est_val', 'est_key', 'trial'], rsuffix='_!mbase')
        data = data.join(
            dstd, on=['est_val', 'est_key', 'trial'], rsuffix='_!sbase')
        for area in areas:
            data.loc[:, area] = data.loc[:, area] - \
                data.loc[:, area + '_!mbase']
            data.loc[:, area] = data.loc[:, area] / \
                data.loc[:, area + '_!sbase']
    return data.loc[:, target_cols]


def ensure_iter(input):
    if isinstance(input, basestring):
        yield input
    else:
        try:
            for item in input:
                yield item
        except TypeError:
            yield input
