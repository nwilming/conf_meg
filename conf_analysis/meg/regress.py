#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Decoding analyses for conf_meg data.

4.  Signed confidence and choice decoder: Same as MIDC and MDDC but with
    confidence folded into the responses (-2, -1, 1, 2)
5.  Unsigned confidence decoder: Same as MIDC and MDDC but decode
    confidence only.
"""

import logging
import os
import numpy as np
import pandas as pd

from os.path import join
from glob import glob

from pymeg import aggregate_sr as asr
from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing
from conf_analysis.meg import decoding_analysis as da

from sklearn import linear_model, discriminant_analysis, svm
from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint as sp_randint
import pickle


from joblib import Memory

if "TMPDIR" in os.environ.keys():
    memory = Memory(cachedir=os.environ["PYMEG_CACHE_DIR"])
    inpath = "/nfs/nwilming/MEG/sr_labeled/aggs"
    outpath = "/nfs/nwilming/MEG/sr_decoding/"
elif "RRZ_LOCAL_TMPDIR" in os.environ.keys():
    tmpdir = os.environ["RRZ_LOCAL_TMPDIR"]
    outpath = "/work/faty014/MEG/sr_labeled/aggs/"
    outpath = "/work/faty014/MEG/sr_decoding/"
    memory = Memory(cachedir=tmpdir)
else:
    inpath = "/home/nwilming/conf_meg/sr_labeled/aggs"
    outpath = "/home/nwilming/conf_meg/sr_decoding"
    memory = Memory(cachedir=metadata.cachedir)

n_jobs = 1


def shuffle(data):
    return data[np.random.permutation(np.arange(len(data)))]

def eval_areas(cache_key, motor_latency=1.2, latency_stim=np.arange(-0.1, 0.35, 1 / 60.0)):
    scores = []
    for area in ['JWG_M1', 'JWG_IPS_PCeS', 'JWG_aIPS', 'vfcFEF']:
        sc, _, _, _ = eval_stim_latency(area, motor_latency=motor_latency, latency_stim=latency_stim, 
            motor_area=area)
        scores.append(sc)
    scores = pd.concat(scores, 0)
    pickle.dump(
        {
            "scores": scores,           
        },
        open(
            "/home/nwilming/conf_analysis/results/%s_Xarea_stim_latency.pickle" % cache_key,
            "wb",
        ),
    )


def eval_stim_latency(
    cache_key,
    motor_latency=1.4,
    baseline=False,
    latency_stim=np.arange(-0.1, 0.4, 1 / 60.0),
    n_jobs=12,
    motor_area='JWG_M1',
):
    from joblib import Parallel, delayed
    import pickle

    args = list(
        delayed(eval_all_subs)(motor_latency, i, baseline, motor_area) for i in latency_stim
    )
    results = Parallel(n_jobs=8)(args)
    sc = []
    weights = {}
    shuff_weights = {}
    iweights = {}
    for j, i in enumerate(latency_stim):
        results[j][0].loc[:, "latency"] = i
        sc.append(results[j][0])
        weights[i] = results[j][1]
        shuff_weights[i] = results[j][2]
        iweights[i] = results[j][3]
    scores = pd.concat(sc, 0)
    pickle.dump(
        {
            "scores": scores,
            "weights": weights,
            "shuffled_weights": shuff_weights,
            "1smp_weights": iweights,
        },
        open(
            "/home/nwilming/conf_analysis/results/%s_stim_latency.pickle" % cache_key,
            "wb",
        ),
    )
    return scores, weights, shuff_weights, iweights


def get_cache_key(cache_key):
    o = pickle.load(
        open(
            "/home/nwilming/conf_analysis/results/%s_stim_latency.pickle" % cache_key,
            "rb",
        )
    )
    return o["scores"], o["weights"], o["shuffled_weights"], o["1smp_weights"]


def eval_all_subs(latency_motor=1.4, latency_stim=0.18, baseline_correct=False, 
    motor_area='JWG_M1'):
    scores = []
    shuffled_scores = []
    iscores = []
    weights = []
    sweights = []
    iweights = []
    print("test")
    for subject in range(1, 16):
        s, w, ss, sw, si, iw = eval_coupling(
            subject,
            latency_motor=latency_motor,
            latency_stim=latency_stim,
            baseline_correct=baseline_correct,
            motor_area=motor_area,
        )
        scores.append(s)
        weights.append(w)
        shuffled_scores.append(ss)
        sweights.append(sw)
        iscores.append(si)
        iweights.append(iw)
    scores = pd.DataFrame(
        {"corr": scores, "1smp_corr": iscores, "shuff_corr": shuffled_scores}
    )
    scores.loc[:, "subject"] = np.arange(1, 16)
    scores.loc[:, 'motor_area'] = motor_area
    return scores, weights, sweights, iweights


def weight_to_act(X, w, i):
    w = np.concatenate((w, [i]))
    SXn = np.cov(X.T)
    #w = w[:, np.newaxis]
    return np.dot(SXn, w)

def eval_coupling(
    subject, latency_motor=1.4, latency_stim=0.18, baseline_correct=False,
    motor_area='JWG_M1'
):
    motor, evals = get_motor_prediction(subject, latency_motor, cluster=motor_area)
    print("S %i AUC:" % subject, evals["test_roc_auc"])
    lodds = np.log(motor.loc[:, 0] / motor.loc[:, 1]).values
    X, tpc, freqs = build_design_matrix(
        subject,
        motor.index,
        "vfcPrimary",
        latency_stim,
        add_contrast=False,
        zscore=True,
        freq_bands=[0, 8, 39, 61, 100],
    )
    if baseline_correct:
        base, tpc, freqs = build_design_matrix(
            subject,
            motor.index,
            "vfcPrimary",
            0,
            add_contrast=False,
            zscore=True,
            freq_bands=[0, 8, 39, 61, 100],
        )
        X = X - base

    score, weights, intercept = coupling(lodds, X, n_iter=250, pcdist=sp_randint(5, 40))
    shuffled_score, shuffled_weights, s_intercept = coupling(
        shuffle(lodds), X, n_iter=250, pcdist=sp_randint(5, 40)
    )
    inst_corr, iweights, _ = coupling(lodds, X[:, -2:], n_iter=250, pcdist=None)
    return (score, weight_to_act(X, weights, intercept), 
        shuffled_score, weight_to_act(X, shuffled_weights, s_intercept), 
        inst_corr, iweights)


@memory.cache()
def get_motor_prediction(subject, latency, cluster="JWG_M1"):
    # First load low level averaged stimulus data
    filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, "stimulus")))
    data = asr.delayed_agg(filenames, hemi="Lateralized", cluster=cluster)()
    meta = da.augment_meta(da.preprocessing.get_meta_for_subject(subject, "stimulus"))
    scores = da.midc_decoder(
        meta,
        data,
        cluster,
        latency=latency,
        splitmc=False,
        target_col="response",
        predict=True,
    )
    eval_scores = da.midc_decoder(
        meta,
        data,
        cluster,
        latency=latency,
        splitmc=False,
        target_col="response",
        predict=False,
    )
    return scores, eval_scores


@memory.cache()
def build_design_matrix(
    subject,
    trial_index,
    cluster,
    latency,
    add_contrast=False,
    zscore=True,
    freq_bands=None,
):
    filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, "stimulus")))
    data = asr.delayed_agg(filenames, hemi="Averaged", cluster=cluster)()
    X, time_per_col = prep_low_level_data(
        cluster, data, 0, latency, trial_index, freq_bands=freq_bands
    )
    cols = X.columns.values
    index = X.index.values
    X = X.values

    if add_contrast:
        meta = da.preprocessing.get_meta_for_subject(subject, "stimulus")
        meta.set_index("hash", inplace=True)
        meta = meta.loc[trial_index]
        cvals = np.stack(meta.contrast_probe)
        X = np.hstack([X, cvals])
    if zscore:
        X = (X - X.mean(0)) / X.std(0)
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    return X, time_per_col, cols


def prep_low_level_data(areas, data, peak, latency, trial_index, freq_bands=None):
    lld = []
    times = data.columns.get_level_values("time").values
    time_per_col = []
    for s in np.arange(0, 1, 0.1) + latency:
        # Turn data into (trial X Frequency)
        target_time_point = times[np.argmin(abs(times - s))]
        for a in da.ensure_iter(areas):
            x = pd.pivot_table(
                data.query('cluster=="%s"' % a),
                index="trial",
                columns="freq",
                values=target_time_point,
            )
            if freq_bands is not None:
                x = (
                    x.T.groupby(
                        pd.cut(
                            x.T.index,
                            freq_bands,
                            labels=np.array(freq_bands)[:-1] + np.diff(freq_bands)/2,
                        )
                    )
                    .mean()
                    .T
                )
            lld.append(x.loc[trial_index])
            time_per_col.extend([s] * x.shape[1])
    return pd.concat(lld, 1), time_per_col


def coupling(target, X, n_iter=50, pcdist=sp_randint(5, 40)):
    """
    
    """
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
    from sklearn.metrics import roc_auc_score, mean_squared_error
    from sklearn.decomposition import PCA
    from sklearn.utils import shuffle
    from imblearn.pipeline import Pipeline
    from sklearn.metrics.scorer import make_scorer

    corr_scorer = make_scorer(
        lambda x, y: np.corrcoef(x, y)[0, 1], greater_is_better=True
    )

    classifier = Pipeline(
        [
            # ("Scaling", StandardScaler()),
            #("PCA", PCA(n_components=0.3, svd_solver="full")),
            ("linear_regression", Ridge(fit_intercept=False)),
        ]
    )
    if pcdist is not None:
        classifier = RandomizedSearchCV(
            classifier,
            param_distributions={
                #"PCA__n_components": pcdist,
                "linear_regression__alpha": sp_randint(1, 100000),
            },
            n_iter=n_iter,
            cv=3,
        )
    else:
        classifier = RandomizedSearchCV(
            classifier,
            param_distributions={"linear_regression__alpha": sp_randint(1, 100000)},
            n_iter=n_iter,
            cv=3,
        )
    scores = cross_validate(
        classifier,
        X,
        target,
        cv=3,
        scoring=corr_scorer,
        return_train_score=False,
        return_estimator=True,
        n_jobs=1,
    )

    #coefs = np.stack(
    #    [
    #        np.dot(
    #            o.best_estimator_.steps[-1][1].coef_,
    #            o.best_estimator_.steps[-2][1].components_,
    #        )
    #        for o in scores["estimator"]
    #    ]
    #)
    coefs = np.stack(
        [            
            o.best_estimator_.steps[-1][1].coef_   
               for o in scores["estimator"]
        ]
    )
    coefs = coefs.mean(0)
    return scores["test_score"].mean(), coefs[:-1], coefs[-1]
