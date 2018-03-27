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
from scipy.special import erf
from scipy.interpolate import interp1d


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
def sigmoidal_old(contrast, slope, offset):
    '''
    Sigmoidal transform of contrast
    '''
    y = expit((slope * (contrast - 0.5 + offset))) - 0.5
    y = y / (expit((slope * (1 - 0.5 + offset))) - 0.5)
    return y


@jit(nopython=True)
def sigmoidal(contrast, slope, offset):
    '''
    Sigmoidal transform of contrast
    '''
    y = expit((slope * (contrast - 0.5 + offset))) - 0.5
    return y / (expit((slope * (1 - 0.5))) - 0.5)


def sigmoid_transformer(x, amplitude, offset, a, b):
    '''
    x is in range (-1, 1)
    '''

    b = erf(b)
    if a >= 0:
        y = erf(abs(a) * (x + b))
        y -= y.min()
        y = 2 * ((y / y.max()) - 0.5)
    if a < 0:
        xx = np.linspace(-1, 1, 200)
        y = erf(-a * (xx + b))
        y -= y.min()
        y = 2 * ((y / y.max()) - 0.5)
        spl = interp1d(y, xx, kind='linear')
        y = spl(x)
    return amplitude * y + offset


def spitzer_exp(contrast, kappa, offset):
    '''
    Kappa is parameterized such that 0 is a linear relationship
    Negative kappa is compression
    '''
    term = (2 * (contrast - 0.5)) + offset
    return ((np.abs(term)**kappa) * term / np.abs(term))


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
Linear model with temporal differences and Spitzer et al.
non-linearities: spitzer
The contrast dependence is non-linear and can follow a sigmoid
or inverted sigmoid.
'''


def spitzer_predict(time, contrast, offset=None,
                    latency=0.2, std=0.05,
                    kappa=1.0, sigmoid_offset=0.0,
                    amplitude_parameters=1,
                    diff_latency=0.2, diff_std=0.05,
                    diff_amplitude_parameters=1):
    if len(contrast.shape) < 2:
        raise RuntimeError('Contrast array needs to be of dimension trials:10')
    contrast = spitzer_exp(contrast, kappa, sigmoid_offset)
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


def spitzer_params2dict(time, x, offset=True):
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
        'kappa': x[tl + 2],
        'sigmoid_offset': x[tl + 3],
        'amplitude_parameters': x[tl + 4],
        'diff_latency': x[tl + 5],
        'diff_std': x[tl + 6],
        'diff_amplitude_parameters': x[tl + 7]})
    return params


def spitzer_fit(data, contrast, time, offset=True):
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
                          values=area).loc[:, -0.2:1.3]
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


def subject_sa(subject, F=[40, 45, 50, 55, 60, 65, 70],
               areas=['V1-lh', 'V2-lh', 'V3-lh', 'V1-rh', 'V2-rh', 'V3-rh'],
               window=0.01, remove_overlap=True):
    '''
    Extraxt single trial power estimates.
    '''
    acc_real, acc_pred = [], []
    for f, area in product(F, areas):
        if not subject_fit.is_cached(subject, f, area):
            print 'Skipping', subject, f, area
            continue
        data, contrast, params = subject_fit(subject, f, area)
        sa = sample_aligned(data, contrast, params,
                            window=window, predict_func=sltd_predict,
                            remove_overlap=remove_overlap)

        sa.loc[:, 'subject'] = subject
        sa.loc[:, 'F'] = f
        sa.loc[:, 'area'] = area
        acc_real.append(sa)
        predicted = sltd_predict(data.columns.values, contrast, **params).T
        predicted = pd.DataFrame(
            data, index=np.arange(data.shape[0]), columns=data.columns.values)

        sa = sample_aligned(predicted, contrast, params,
                            window=window, predict_func=sltd_predict,
                            remove_overlap=remove_overlap)
        sa.loc[:, 'subject'] = subject
        sa.loc[:, 'F'] = f
        sa.loc[:, 'area'] = area
        acc_pred.append(sa)
    return pd.concat(acc_real), pd.concat(acc_pred)


def make_all_sub_sa():
    out_sa, out_sp = [], []
    for subject in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        sa, sp = subject_sa(subject)
        out_sa.append(sa)
        out_sp.append(sp)
    return pd.concat(out_sa), pd.concat(out_sp)


def sample_aligned(data, contrast, params, window=0.01, remove_overlap=True,
                   predict_func=sltd_predict):
    '''
    Produce sample aligned power values, but remove effec from previous sample
    '''

    latency = params['latency']
    output = []
    cnt = 0
    for sample, onset in enumerate(np.arange(0, 1, 0.1) + latency):
        if remove_overlap:
            con = contrast.copy()
            con[:, sample] = 0.5
            predicted = predict_func(data.columns.values, con, **params).T
            predicted = pd.DataFrame(
                predicted, columns=data.columns, index=data.index)
            vs = (data - predicted).loc[:, onset -
                                        window:onset + window].mean(1)
        else:
            vs = data.loc[:, onset - window:onset + window].mean(1)
        for trial, cc, d in zip(vs.index.values, contrast[:, sample], vs):
            output.append({'sample': sample, 'power': d,
                           'contrast': cc, 'trial': trial, 'sample_id': cnt})
            cnt += 1
    return pd.DataFrame(output)


def get_average_contrast(sa, by=['subject', 'area', 'F'], centers=np.linspace(0, 1, 10),
                         width=0.2):
    dy = sa.groupby(
        by).apply(lambda x: contrast_integrated_averages(x, centers, width))
    return dy


def contrast_integrated_averages(sa, centers=np.linspace(0.1, 0.9, 5),
                                 width=0.2):
    sa = sa.reset_index()
    rows = []
    contrast = sa.loc[:, 'contrast'].values
    w = width / 2.
    for center in centers:
        idx = ((center - w) < contrast) & (contrast < (center + w))
        r = sa.loc[idx, ('contrast', 'power')].mean()
        r.loc['contrast'] = center
        rows.append(r)
    return pd.concat(rows, 1).T


def example_plot():
    from pylab import plot, subplot, ylim, xlabel, ylabel
    import seaborn as sns
    c = np.linspace(0, 1, 100)
    subplot(1, 3, 1)
    color = sns.color_palette("Blues", n_colors=10)
    for ii, slope in enumerate(np.arange(1, 11, 1)):
        plot(c, sigmoidal(c, slope, 0), color=color[ii])
        ylim([-1, 1])
        xlabel('contrast')
        ylabel(r'$\Delta Power$')
    subplot(1, 3, 2)
    color = sns.color_palette("Greens", n_colors=10)
    for ii, amplitude in enumerate(np.linspace(0, 1, 10)):
        plot(c, amplitude * sigmoidal(c, 2.5, 0), color=color[ii])
        ylim([-1, 1])
        xlabel('contrast')
    subplot(1, 3, 3)
    color = sns.color_palette("Reds", n_colors=10)
    for ii, offset in enumerate(np.linspace(-0.5, 0.5, 10)):
        plot(c, 1 * sigmoidal(c, 5, offset), color=color[ii])
        ylim([-1, 1])
        xlabel('contrast')

    sns.despine(offset=5)


def plot_fit_params(parameters):
    import seaborn as sns
    ps = (parameters
          .set_index(['F', 'area', 'subject', 'hemisphere'])
          .loc[:, ['slope', 'amplitude_parameters', 'sigmoid_offset', 'latency']]
          .stack()
          .reset_index())
    ps.columns = ['F', 'area', 'subject', 'hemisphere', 'parameter', 'value']
    g = sns.FacetGrid(ps, col="hemisphere", row="parameter",
                      hue='area', sharey=False, sharex=True, size=2, aspect=2.5)
    g.map(sns.pointplot, 'F', 'value')
    g.axes[0, 0].set_ylim([0, 11])
    g.axes[0, 1].set_ylim([0, 11])

    g.axes[1, 0].set_ylim([-0.1, 0.6])
    g.axes[1, 1].set_ylim([-0.1, 0.6])

    g.axes[2, 0].set_ylim([-0.4, 0.3])
    g.axes[2, 1].set_ylim([-0.4, 0.3])

    g.axes[3, 0].set_ylim([0.0, 0.3])
    g.axes[3, 1].set_ylim([0.0, 0.3])
'''
Bayesian fits
'''


def t_model(power, contrast):
    '''
    Define a model that predicts contrast from power. Two step approach:
    First fit likelihood to observe a power value based on contrast.

    X contains power values, each colum is a feature (area / Freq)
    c contains contrast values
    '''
    import numpy as np
    # import theano.tensor as tt
    # import pymc3 as pm

    with pm.Model() as model:

        # Now define contrast dependence.
        l = pm.Gamma("l", alpha=2, beta=1)
        eta = pm.HalfCauchy("eta", beta=5)

        cov = eta**2 * pm.gp.cov.Matern52(1, l)
        gp = pm.gp.Latent(cov_func=cov)

        f = gp.prior("f", X=contrast)

        sigma = pm.HalfCauchy("sigma", beta=5)
        nu = pm.Gamma("nu", alpha=2, beta=0.1)
        y_ = pm.StudentT("power", mu=f, lam=1.0 /
                         sigma, nu=nu, observed=power)
        trace = pm.sample(1000)
        return trace

'''
Plots
'''


def plot_sample_aligned_responses(sa, area='V1-lh'):
    '''
    Plot sample aligned responses
    '''
    import pylab as plt
    # 1 Plot averaged gp response as a function of contrast and frequency
    dreal = get_average_contrast(sa, by=['subject'], area=area)
    for sub, d in dreal.groupby('subject'):
        plt.plot(d.contrast, d.power, label=sub)
    # 2 Plot predicted gp response as a function of contrast and frequency


'''
CRFs
'''


def H(x, m, p, q, c):
    c = float(c)

    k = (m * (x**p)) / (x**(p + q) + c**(p + q))

    if k[-1] < k.max():
        dk = dH(x, m, p, q, c)
        norm = k[np.argmin(abs(dk))]
        k = k / norm
    else:
        k = k / k.max()
    return m * k - m / 2.


def dH(x, m, p, q, c):
    z = m * x**(p - 1) * (p * (c**(p + q)) - q * (x**(p + q)))
    t = (c**(p + q) + x**(p + q))**2.
    return z / t


'''
Precomputing
'''


def foo(x):
    print x


def precompute_fits():
    from pymeg import parallel
    subs = range(1, 16)
    funcs = ['stld']
    areas = ['V1-lh', 'V2-lh', 'V3-lh', 'V1-rh', 'V2-rh', 'V3-rh']
    from itertools import product
    ids = []
    for sub, area in product(subs, areas):
        F = [40, 45, 50, 55, 60, 65, 70]
        for f in F:
            if not subject_fit.is_cached(sub, f, area):
                ids.append(parallel.pmap(subject_fit, [[sub, f, area]]))
    return ids
