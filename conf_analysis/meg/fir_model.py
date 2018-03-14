'''
This module models gamma band responses to individual samples by assuming that
each response follows a parameterized function (normals for now). The
amplitude of each response is assumed to be modulated by contrast.
'''

from itertools import product
import numpy as np
import pandas as pd

from ..behavior import individual_sample_model as ism
from . import srplots
from numba import jit, vectorize
from scipy.optimize import least_squares

from conf_analysis.behavior import metadata
from joblib import Memory

memory = Memory(cachedir=metadata.cachedir)


@vectorize(['f8(f8)', 'f4(f4)'])
def expit(x):
    if x > 0:
        x = np.exp(x)
        return x / (1 + x)
    else:
        return 1 / (1 + np.exp(-x))


@jit
def gauss(x, mean, sigma, amplitude=1):
    return (np.exp(-(x - mean)**2 / sigma).reshape((-1, 1))
            * amplitude.reshape((1, -1)))


@jit
def power_transformer(x, a):
    '''
    x is in range (0, 1), maps to (0, 1)
    a is in range(-inf, inf)
    '''
    y = x**np.exp(a)
    return y


@jit(nopython=True)
def sigmoidal(contrast, slope, offset):
    '''
    Sigmoidal transform of contrast
    '''
    y = expit((slope * (contrast - 0.5 + offset))) - 0.5
    y = y / (expit((slope * (1 - 0.5 + offset))) - 0.5)
    return y


def fit(predict, data, x0, bounds):
    err = (lambda x:
           (data - predict(x)).ravel())

    return least_squares(err, x0, loss='soft_l1', bounds=bounds)


'''
Linear model with temporal differences: lmtd
Contrast dependence is simply linear!
'''


@jit(nopython=True)
def lmtd_predict(time, contrast, offset=None,
                 latency=0.2, std=0.05,
                 amplitude_parameters=1,
                 diff_latency=0.2, diff_std=0.05,
                 diff_amplitude_parameters=1):
    if len(contrast.shape) < 2:
        raise RuntimeError('Contrast array needs to be of dimension trials:10')
    output = time.reshape((-1, 1)) * np.zeros((1, contrast.shape[0]))

    for ii, onset in enumerate(np.arange(0, 1, 0.1)):
        output += gauss(time, onset + latency, std,
                        contrast[:, ii] * amplitude_parameters)

    cdiff = np.diff(contrast, 1)
    onsets = np.arange(0.1, 1, 0.1)
    for ii, onset in enumerate(onsets):
        output += gauss(time, onset + latency + diff_latency, diff_std,
                        cdiff[:, ii] * diff_amplitude_parameters)
    return output + offset.reshape((-1, 1))


def lmtd_params2dict(time, x):
    tl = len(time)
    return {
        'offset': x[0:tl],
        'latency': x[tl],
        'std': x[tl + 1],
        'amplitude_parameters': x[tl + 2],
        'diff_latency': x[tl + 3],
        'diff_std': x[tl + 4],
        'diff_amplitude_parameters': x[tl + 5]}


def lmtd_fit(data, contrast, time):
    '''
    Use non-linear least squares to find parameters.
    '''
    x0 = np.concatenate((data.mean(1), [0.15, 0.001, 1, 0.001, 0.001, .5]))

    bounds = ([-np.inf] * len(time) + [0.0, 0.0, -np.inf, -0.1, 0.0, -np.inf],
              [+np.inf] * len(time) + [0.5, 0.1, +np.inf, +0.1, 0.1, +np.inf])

    yp = lambda x: lmtd_predict(time, contrast, **lmtd_params2dict(time, x))
    out = fit(yp, data, x0, bounds)
    return lmtd_params2dict(time, out['x'])

'''
Linear model with temporal differences and transformed 
contrast: nltd
Contrast dependence is non-linear!
'''


@jit(nopython=True)
def nltd_predict(time, contrast, offset=None,
                 latency=0.2, std=0.05,
                 non_linearity=1,
                 amplitude_parameters=1,
                 diff_latency=0.2, diff_std=0.05,
                 diff_amplitude_parameters=1, **kw):
    if len(contrast.shape) < 2:
        raise RuntimeError('Contrast array needs to be of dimension trials:10')
    output = time.reshape((-1, 1)) * np.zeros((1, contrast.shape[0]))

    contrast = power_transformer(contrast, non_linearity) - \
        power_transformer(0.5, non_linearity)  # Now contrast of 0.5 -> 0

    for ii, onset in enumerate(np.arange(0, 1, 0.1)):
        output += gauss(time, onset + latency, std,
                        contrast[:, ii] * amplitude_parameters)

    cdiff = np.diff(contrast, 1)
    onsets = np.arange(0.1, 1, 0.1)
    for ii, onset in enumerate(onsets):
        output += gauss(time, onset + latency + diff_latency, diff_std,
                        cdiff[:, ii] * diff_amplitude_parameters)

    return output + offset.reshape((-1, 1))


def nltd_params2dict(time, x):
    tl = len(time)
    return {
        'offset': x[0:tl],
        'latency': x[tl],
        'std': x[tl + 1],
        'non_linearity': x[tl + 2],
        'amplitude_parameters': x[tl + 3],
        'diff_latency': x[tl + 4],
        'diff_std': x[tl + 5],
        'diff_amplitude_parameters': x[tl + 6]}


def nltd_fit(data, contrast, time):
    '''
    Use non-linear least squares to find parameters.
    '''
    from scipy.optimize import least_squares

    x0 = np.concatenate(
        (data.mean(1), [0.15, 0.001, 1., 0.0,  0.001, 0.001, .5]))

    bounds = ([-np.inf] * len(time) +
              [0.0, 0.0, -1, -100, -0.1, 0.0, -np.inf],
              [+np.inf] * len(time) +
              [0.5, 0.1, +1, +100, +0.1, 0.1, +np.inf])

    yp = lambda x: nltd_predict(time, contrast, **nltd_params2dict(time, x))
    err = (lambda x:
           (data - yp(x)).ravel())

    out = least_squares(err, x0, loss='soft_l1', bounds=bounds)
    return nltd_params2dict(time, out['x'])


'''
Linear model with temporal differences and sigmoid 
contrast transform: sltd
The contrast dependence is non-linear.
'''


#@jit(nopython=True)
def sltd_predict(time, contrast, offset=None,
                 latency=0.2, std=0.05,
                 slope=1.0, sigmoid_offset=0.0,
                 amplitude_parameters=1,
                 diff_latency=0.2, diff_std=0.05,
                 diff_amplitude_parameters=1):
    if len(contrast.shape) < 2:
        raise RuntimeError('Contrast array needs to be of dimension trials:10')
    contrast = sigmoidal(contrast, slope, sigmoid_offset)
    output = time.reshape((-1, 1)) * np.zeros((1, contrast.shape[0]))

    for ii, onset in enumerate(np.arange(0, 1, 0.1)):
        output += gauss(time, onset + latency, std,
                        contrast[:, ii] * amplitude_parameters)

    cdiff = np.diff(contrast, 1)
    onsets = np.arange(0.1, 1, 0.1)
    for ii, onset in enumerate(onsets):
        output += gauss(time, onset + latency + diff_latency, diff_std,
                        cdiff[:, ii] * diff_amplitude_parameters)
    if offset is not None:
        output += offset.reshape((-1, 1))
    return output


def sltd_params2dict(time, x, offset=True):
    params = {}
    if len(x) != int(offset) * len(time) + 8:
        raise RuntimeError(
            'Number of parameters is wrong. Is %i, should be %i' % (len(x), len(time) + 8))
    if offset:

        tl = len(time)
        params['offset'] = x[0:tl]
    else:
        tl = 0
        params['offset'] = None

    params.update({
        'latency': x[tl],
        'std': x[tl + 1],
        'slope': x[tl + 2],
        'sigmoid_offset': x[tl + 3],
        'amplitude_parameters': x[tl + 4],
        'diff_latency': x[tl + 5],
        'diff_std': x[tl + 6],
        'diff_amplitude_parameters': x[tl + 7]})
    return params


def sltd_fit(data, contrast, time, offset=True):
    '''
    Use non-linear least squares to find parameters.
    '''
    from scipy.optimize import least_squares

    if offset:
        x0 = np.concatenate(
            (data.mean(1), [0.15, 0.001, 1., 0.0,  0.0,  0.001, 0.001, .5]))
        # Offset, latency, std, slope, sigmoid_offset, ...
        bounds = ([-np.inf] * len(time) +
                  [0.0, 0.0,   0, -0.5, -np.inf, -0.1, 0.0, -np.inf],
                  [+np.inf] * len(time) +
                  [0.5, 0.1, +10, +0.5, +np.inf, +0.1, 0.1, +np.inf])
    else:
        x0 = np.array([0.15, 0.001, 1., 0.0,  0.0,  0.001, 0.001, .5])
        data = data - data.mean(0)
        bounds = ([0.0, 0.0, -10, -0.5, -np.inf, -0.1, 0.0, -np.inf],
                  [0.5, 0.1, +10, +0.5, +np.inf, +0.1, 0.1, +np.inf])

    yp = lambda x: sltd_predict(
        time, contrast, **sltd_params2dict(time, x, offset=offset))
    err = (lambda x:
           (data - yp(x)).ravel())

    out = least_squares(err, x0, loss='soft_l1', bounds=bounds)
    return sltd_params2dict(time, out['x'], offset=offset)


'''
Analysis functions
'''


@memory.cache
def make_sub_data(subject, area, F=55, log=True):
    df, meta = srplots.get_power(subject, decim=3, F=F)
    cvals = ism.get_contrast(meta)
    data = make_data(df, area=area, log=log)
    return data, cvals


def make_data(df, area='V1-lh', log=False):
    data = pd.pivot_table(df, index='trial', columns='time',
                          values='V1-lh').loc[:, -0.2:1.3]
    if log:
        data = np.log(data)
    base = data.loc[:, -0.2:0].mean(1)
    bases = data.loc[:, -0.2:0].std(1)
    return data.subtract(base, 0).div(bases, 0)


@memory.cache
def subject_fit(subject, F, area, fit_func='stld'):
    fit_map = {'stld': sltd_fit, 'nltd': nltd_fit, 'lmtd': lmtd_fit}
    data, contrast = make_sub_data(subject, area, F=F)
    time = data.columns.values
    params = fit_map[fit_func](data.values.T, contrast, time)
    return data, contrast, params


@memory.cache
def subject_sa(subject, F=[40, 45, 50, 55, 60, 65, 70],
               areas=['V1-lh', 'V2-lh', 'V3-lh', 'V1-rh', 'V2-rh', 'V3-rh'],
               window=0.01):
    acc_real, acc_pred = [], []
    for f, area in product(F, areas):
        data, contrast, params = subject_fit(subject, f, area)
        print 'Loaded subject fits'
        sa = sample_aligned_rm(data, contrast, params,
                               window=window, predict_func=sltd_predict)
        sa.loc[:, 'subject'] = subject
        sa.loc[:, 'F'] = f
        sa.loc[:, 'area'] = area
        acc_real.append(sa)
        predicted = sltd_predict(data.columns.values, contrast, **params).T
        predicted = pd.DataFrame(
            data, index=np.arange(data.shape[0]), columns=data.columns.values)
        sa = sample_aligned_rm(predicted, contrast, params,
                               window=window, predict_func=sltd_predict)
        sa.loc[:, 'subject'] = subject
        sa.loc[:, 'F'] = f
        sa.loc[:, 'area'] = area
        acc_pred.append(sa)
    return pd.concat(acc_real), pd.concat(acc_pred)


def sample_aligned_rm(data, contrast, params, window=0.01, predict_func=nltd_predict):
    '''
    Produce sample aligned power values, but remove effec from previous sample
    '''
    latency = params['latency']
    output = []
    for sample, onset in enumerate(np.arange(0, 1, 0.1) + latency):
        con = contrast.copy()
        con[:, sample] = 0.5
        predicted = predict_func(data.columns.values, con, **params).T
        predicted = pd.DataFrame(
            predicted, columns=data.columns, index=data.index)
        vs = (data - predicted).loc[:, onset - window:onset + window].mean(1)
        for trial, cc, d in zip(vs.index.values, contrast[:, sample], vs):
            output.append({'sample': sample, 'power': d,
                           'contrast': cc, 'trial': trial})
    return pd.DataFrame(output)


def sample_aligned(data, contrast, latency, window=0.01):
    output = []
    for sample, onset in enumerate(np.arange(0, 1, 0.1) + latency):
        vs = data.loc[:, onset - window:onset + window].mean(1)
        for cc, d in zip(contrast[:, sample], vs):
            output.append({'sample': sample, 'power': d, 'contrast': cc})
    return pd.DataFrame(output)


def get_average_contrast(contrast, params, time,  data=None,
                         edges=np.linspace(0.1, 0.9, 8),
                         predict_func=lmtd_predict,
                         remove_overlap=False):
    if data is None:
        data = predict_func(time, contrast, **params).T
        data = pd.DataFrame(data, index=np.arange(data.shape[0]), columns=time)
    if remove_overlap:
        dy = sample_aligned_rm(data, contrast, params,
                               predict_func=predict_func)
    else:
        dy = sample_aligned(data, contrast, params['latency'])
    dy = dy.groupby(['sample', pd.cut(dy.contrast, edges)]).mean()
    dy.index.names = ['sample', 'cbins']
    return dy


'''
Plots
'''


def plot_sample_aligned_responses():
    '''
    Plot sample aligned responses
    '''
    pass


'''
Precomputing
'''


def precompute_fits():
    from pymeg import parallel
    subs = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    F = [40]#, 45, 50, 55, 60, 65, 70]
    funcs = ['stld']
    areas = ['V1-lh']#, 'V2-lh', 'V3-lh', 'V1-rh', 'V2-rh', 'V3-rh']
    from itertools import product
    for sub, f, area in product(subs, F, areas):
        #parallel.pmap(make_sub_data, [[sub, area, f]])        
        parallel.pmap(subject_fit, [[sub, f, area]])
    '''
    for sub in subs:
        paralle.pmap(subject_sa, [[sub, [40, 45, 50, 55, 60, 65, 70],
                     ['V1-lh', 'V2-lh', 'V3-lh', 'V1-rh', 'V2-rh', 'V3-rh'],
                     0.01]])
    '''
