import pandas as pd
from numpy import *
import numpy as np
import seaborn as sns
sns.set_style('ticks')
import glob
from scipy.io import loadmat
import statsmodels.api as sm
from sklearn import linear_model
import patsy
import time
from scipy.stats import norm
from os.path import join
import metadata, keymap

from joblib import Memory
from os.path import basename, join

import matplotlib

memory = Memory(cachedir=metadata.cachedir, verbose=0)

def load_jw():
    df = pd.read_csv('/Users/nwilming/u/conf_analysis/jw_yes_no_task_data.csv')
    df.columns = [u'Unnamed: 0', u'blinks_nr', u'confidence', u'contrast', u'correct',
       u'noise_redraw', u'present', u'pupil_b', u'pupil_d', u'choice_rt', u'block_num',
       u'sacs_nr', u'session_num', u'staircase', u'snum', u'trial', u'response']
    sub_map = dict((k, v) for v, k in enumerate(unique(df.snum)))
    df.loc[:, 'snum'] = df.snum.replace(sub_map)
    #def foo(x):
    #    x.loc[:, 'trial'] = arange(len(x))+1
    #    return x
    #df = df.groupby(['session_num', 'block_num', 'snum']).apply(foo)
    return df.drop(u'Unnamed: 0', axis=1)


@memory.cache
def load_data():
    '''
    Load behavioral matlab data files.

    Some files need special treatment, e.g. because subjects confused response keys.
    '''
    files = glob.glob(join(metadata.behavioral_path, 'S*/s*.mat'))
    files += glob.glob(join(metadata.behavioral_path, 's*/s*.mat'))
    files += glob.glob(join(metadata.behavioral_path, 's*/S*.mat'))
    files += glob.glob(join(metadata.behavioral_path, 'S*/S*.mat'))
    files = unique(files)

    dfs = []
    def unbox(l):
        if len(l)==1:
            return l[0]
        else:
            return l

    for f in files:
        data = loadmat(f)
        m = data['session_struct']['results'][0,0].ravel()
        df = pd.DataFrame([dict((k, unbox(i[k].ravel())) for k in m.dtype.fields) for i in m])
        subject = int(f.split('/')[5][1:3])
        df['snum'] = subject
        df['trial'] = arange(len(df))
        dfs.append(df)
    data = pd.concat(dfs).reset_index()
    day = [int(time.strftime('%Y%m%d', time.strptime(k, '%Y%m%dT%H%M%S'))) for k in data.session.values]
    data['day'] = day
    data['mc'] = array([mean(k) for k in data.contrast_probe.values])
    data['stdc'] = array([std(k) for k in data.contrast_probe.values])
    data['R'] = data.response.copy()
    data.loc[data.response==-1, 'R'] = 0

    def session_num(data):
        lt = dict((k, i) for i,k in enumerate(sort(unique(data.day))))
        data.loc[:, 'session_num'] = data.day
        data.session_num = data.session_num.replace(lt)
        return data

    def block_num(data):
        lt = dict((k, i) for i,k in enumerate(sort(unique(data.session.astype('str')))))
        data.loc[:, 'block_num'] = data.session.astype('str')
        data.block_num = data.block_num.replace(lt)
        return data

    def contrast_block_mean(data):
        con = abs(vstack(data.contrast_probe)-0.5)
        m = mean(con)
        data.loc[:, 'contrast_block_mean'] = m
        return data


    data = data.groupby('snum').apply(session_num)
    data = data.groupby(['snum', 'session_num']).apply(block_num)
    data = data.groupby(['snum', 'session_num', 'block_num']).apply(contrast_block_mean)

    data.loc[:, 'hash'] =  [keymap.hash(x) for x in data.loc[:, ('day', 'snum', 'block_num', 'trial')].values]
    assert len(np.unique(data.loc[:, 'hash'])) == len(data.loc[:, 'hash'])
    return data


def data_cleanup(data):
    id_2nd_high = ((data.snum==3) & (data.response == 1) & (data.confidence==2) &
                  ((data.day == 20151207) | (data.day == 20151208)))
    id_2nd_low =  ((data.snum==3) & (data.response == 1) & (data.confidence==1) &
                  ((data.day == 20151207) | (data.day == 20151208)))
    data.loc[id_2nd_high, 'confidence'] = 1
    data.loc[id_2nd_low, 'confidence'] = 2
    id_1st_high = ((data.snum==11) & (data.response == -1) & (data.confidence==2) &
              ((data.day == 20160222)))
    id_1st_low =  ((data.snum==11) & (data.response == -1) & (data.confidence==1) &
              ((data.day == 20160222)))
    data.loc[id_1st_high, 'confidence'] = 1
    data.loc[id_1st_low, 'confidence'] = 2
    return data


def phi(x):
    '''
    Inverse cumulative normal
    '''
    return norm.ppf(x)


def tbl(data, field='response'):
    '''
    Compute hit, false alarm, miss and correct rejection rate.
    '''
    hit = sum((data[field] == 1) & (data.side==1))
    fa = sum((data[field] == 1) & (data.side==-1))
    miss = sum((data[field] == -1) & (data.side==1))
    cr = sum((data[field] == -1) & (data.side==-1))
    Na = float(hit + miss)+1.
    Nb = float(fa + cr)+1.
    return hit/Na, fa/Nb, miss/Na, cr/Nb


def dp(data, field='response'):
    '''
    compute d'
    '''
    hit, fa, _, _ = tbl(data, field=field)
    if hit==1 or fa==1:
        raise RuntimeError('Too good or too bad')
    return phi(hit) - phi(fa)


def crit(data, field='response'):
    '''
    compute criterion.
    '''
    hit, fa, _, _ = tbl(data, field=field)
    return -.5 *(phi(hit) + phi(fa))


def acc(data, field='correct'):
    '''
    compute accuracy
    '''
    return data[field].mean()


def pk(df, gs=matplotlib.gridspec.GridSpec(1,3), row=0):
    subplot(gs[row,0])
    conf_kernels(df)
    plot([0.5, 9.5], [0.5, 0.5], 'k--')
    ylabel('Contrast of 2nd grating')
    subplot(gs[row, 1])
    conf_kernels(df[df.correct==1])
    plot([0.5, 9.5], [0.5, 0.5], 'k--')
    xticks([])
    yticks([])
    xlabel('')
    subplot(gs[row, 2])
    conf_kernels(df[df.correct==0])
    plot([0.5, 9.5], [0.5, 0.5], 'k--')
    xticks([])
    yticks([])
    xlabel('')
    legend(bbox_to_anchor=(1.001, 1), loc=2, borderaxespad=0.)


def bootstrap(v, n, N, func=nanmean, alpha=.05):
    '''
    Bootstrap values in func(v[samples]), by drawing n samples N times.
    Returns CI whose width is specified by alpha.
    '''
    r = []
    for i in range(N):
        id_rs = np.random.randint(0, len(v), size = (n,))
        r.append(func(v[id_rs]))
    return np.prctile(r, [(alpha*100)/2, 50, 100 - (alpha*100)/2])


def conf_kernels(df, alpha=1, rm_mean=False, label=True, err_band=False):
    legend_labels = {(-1., 2.): 'Yes, the 1st', (-1., 1.): 'Maybe, the 1st',
                     ( 1., 2.): 'Yes, the 2nd', (1., 1.): 'Maybe, the 2nd'}
    colors = dict((k, c) for k,c in zip(((-1., 2.), (-1., 1.), ( 1., 2.), (1., 1.)), sns.color_palette(n_colors=4)))

    for cond, group in df.groupby(['response', 'confidence']):
        kernel = vstack(group['contrast_probe'].values)
        # trials x time
        x = arange(0.5, 9.6, 1)
        if rm_mean:
            tmean = kernel.mean(1)
            kernel = kernel - tmean[:, np.newaxis]
        if err_band:
            band = array(
                        [bootstrap(kernel[:, i], kernel.shape[0], 1000)
                            for i in range(kernel.shape[1])])
            fill_between(x, band[:, 0], band[:, 2], color=colors[cond], alpha=alpha)
        if label:
            plot(x, kernel.mean(0), label=legend_labels[cond], color=colors[cond], alpha=alpha)
        else:
            plot(x, kernel.mean(0), color=colors[cond], alpha=alpha)

    #ylim([0.35, .65])
    #yticks([0.35, 0.5, 0.65])
    xlabel('time')
    sns.despine()
    #legend()


def asfuncof(xval, data, bins=linspace(1, 99, 12), aggregate=np.mean, remove_outlier=True):
    low, high = np.percentile(xval, [1, 99])
    idx = (low<xval) & (xval<high)
    xval = xval[idx]
    data = data[idx]
    edges = np.percentile(xval, bins)
    cfrac = []
    centers = []
    frac = []
    for low, high in zip(edges[:-1], edges[1:]):
        d = data[(low<xval) & (xval<=high)]
        if len(d) < 5:
            frac.append(np.nan)
        else:
            frac.append(aggregate(d))
        centers.append(nanmean([low, high]))
    return centers, frac


def fit_logistic(df, formula, summary=True):
    y,X = patsy.dmatrices(formula, df, return_type='dataframe')
    log_res = sm.GLM(y, X, family=sm.families.Binomial())
    results = log_res.fit(disp=False)
    if summary:
        print results.summary()
    return log_res, results


def fit_pmetric(df, features=['contrast'], targetname='response'):
    log_res = linear_model.LogisticRegression()
    features = vstack([df[f].values for f in features]).T
    target = df[targetname].values
    log_res.fit(features, target)
    accuracy = log_res.score(features, target)
    print accuracy
    log_res.featnames = features
    log_res.targetname = target
    return log_res


def plot_model(df, model, bins=[linspace(0,.25,100), linspace(0,1,100)],
            hyperplane_only=False, alpha=1, cmap=None):
    C, M = meshgrid(*bins)
    resp1 = histogram2d(df[df.response==1].stdc.values, df[df.response==1].mc.values,
        bins=bins)[0] +1
    resp1[resp1==1] = nan
    resp2 = histogram2d(df[df.response==-1].stdc.values, df[df.response==-1].mc.values,
        bins=bins)[0] +1
    resp2[resp2==1] = nan
    resp1 = resp1.astype(float)/nansum(resp1)
    resp2 = resp2.astype(float)/nansum(resp2)

    p = model.predict(vstack([M.ravel(), C.ravel(), 0*ones(M.shape).ravel()]).T)
    p = p.reshape(M.shape)

    decision = lambda x: -(model.params.mc*x+ model.params.Intercept)/model.params.stdc
    if not hyperplane_only:
        plane = log(resp1/resp2)
        plane[plane==1] = nan
        pcolormesh(bins[1], bins[0], np.ma.masked_invalid(plane), cmap=cmap, vmin=-2.4, vmax=2.4)

    mind, maxd = xlim()
    ylim(bins[0][0], bins[0][-1])
    xlim(bins[1][0], bins[1][-1])
    plot([mind, maxd], [decision(mind), decision(maxd)], 'k', lw=2, alpha=alpha)
    plot([0, 0], [bins[0][0], bins[0][-1]], 'k--', lw=2)
    #ylim([0, 0.25])
    #xlim([0, 1])
    #contour(bins[1], bins[0], p.T, [0.5])
    return bins, p


# Compute kernel data frame
def get_pk(data, contrast_mean=0.5, response_field='response'):
    '''
    Converts data to a data frame that is long form for different contrast probes.
    I.e. indexed by trial, time and whether contrast was for chosen or non-chosen option.

    '''
    dr1 = data.query('%s==1'%response_field)
    dr2 = data.query('%s==-1'%response_field)

    # Subtract QUEST mean from trials.
    con_select2nd = (vstack(dr1.contrast_probe) - contrast_mean
                        - (dr1.contrast_block_mean * dr1.side)[:, newaxis])
    con_select1st = (vstack(dr2.contrast_probe) - contrast_mean
                        - (dr2.contrast_block_mean * dr2.side)[:, newaxis])

    sel = vstack((con_select2nd, 0*con_select1st))
    nsel = vstack((con_select1st, 0*con_select2nd))


    sel = pd.DataFrame(sel)
    sel.index.name='trial'
    sel.columns.name='time'
    sel =  sel.unstack().reset_index()
    sel['optidx'] = 1
    sel = sel.set_index(['time', 'optidx', 'trial'])

    nsel = pd.DataFrame(nsel)
    nsel.index.name='trial'
    nsel.columns.name='time'
    nsel = nsel.unstack().reset_index()
    nsel['optidx'] = 0
    nsel = nsel.set_index(['time', 'optidx', 'trial'])
    df = pd.concat((sel, nsel))
    df.rename(columns={0:'contrast'}, inplace=True)
    return df


def get_decision_kernel(data, contrast_mean=0.5, response_field='response'):
    kernels = []

    kernel = (data.groupby(['snum'])
         .apply(lambda x: get_pk(x, contrast_mean=contrast_mean, response_field=response_field))
         .groupby(level=['snum', 'time', 'optidx']).mean()
         .reset_index())

    kernel_diff = (data.groupby(['snum'])
         .apply(lambda x: get_pk(x, contrast_mean=contrast_mean, response_field=response_field))
         .groupby(level=['snum', 'time'])
              .apply(lambda x: x.query('optidx==1').mean() + x.query('optidx==0').mean())
         .reset_index())

    kernel_diff['optidx'] = 3
    kernel = pd.concat((kernel, kernel_diff))

    condition = kernel.optidx.astype('category')
    condition = condition.cat.rename_categories([r'$E_N$', r'$E_S$', r'$E_D = E_S + E_N$'])
    kernel['Kernel'] = condition
    return kernel


def get_confidence_kernels(data, confidence_field='confidence', response_field='response'):
    kernel = (data.groupby([confidence_field, 'snum'])
        .apply(lambda x: get_pk(x, contrast_mean=0, response_field=response_field))
        .groupby(level=[confidence_field, 'optidx', 'snum', 'time']).mean()
        .reset_index())
    condition = (kernel[confidence_field]*(kernel.optidx-0.5))
    condition = pd.Categorical(condition, categories=[-1, -0.5, 0.5, 1])

    condition = condition.rename_categories([r'$E_{N}^{High}$',
                                                 r'$E_{N}^{Low}$',
                                                 r'$E_{S}^{Low}$',
                                                 r'$E_{S}^{High}$'])
    kernel['Kernel'] = condition.astype(str)
    return kernel


def get_confidence_kernel(data, confidence_field='confidence', response_field='response'):
    kernel = (data.groupby([confidence_field, 'snum'])
         .apply(lambda x: get_pk(x, contrast_mean=0, response_field=response_field))
         .groupby(level=['optidx', 'snum', 'time'])
              .apply(lambda x: x.query('%s==2'%confidence_field).mean()
                              -x.query('%s==1'%confidence_field).mean())
         .reset_index())

    df = lambda x: x.query('optidx==1').mean()+x.query('optidx==0').mean()

    kernel_diff = (kernel.set_index(['snum', 'time', 'optidx'])
                         .groupby(level=['snum', 'time']).apply(df)).reset_index()
    kernel_diff['optidx'] = 3
    kd = pd.concat((kernel, kernel_diff))
    condition = pd.Categorical(kd.optidx, categories=[0, 1, 3])

    condition = condition.rename_categories([ r'$E_{N}^{conf}$',
                                              r'$E_{S}^{conf}$',
                                              r'$E^{conf}$'])
    kd['Kernel'] = condition.astype(str)
    return kd


def plot_kernel(kernel, colors, legend=True, trim=True):
    g = sns.tsplot(time='time', unit='snum', value='contrast', condition='Kernel',
               data=kernel, ci=95, color=colors, legend=legend)

    plot([0, 9], [0, 0], lw=1, color='k', alpha=0.5)
    yticks([-0.2, 0, 0.2])
    xlim([-0.5, 9.25])
    xlabel('Sample #')
    sns.despine(trim=True, ax=gca())
    return g
