import pandas as pd
from pylab import *
import seaborn as sns
sns.set_style('ticks')
import glob
from scipy.io import loadmat
import statsmodels.api as sm
from sklearn import linear_model
import patsy
import time

def load_data():
    '''
    Load behavioral matlab data files.

    Some files need special treatment, e.g. because subjects confused response keys.
    '''
    files = glob.glob('/Users/nwilming/u/conf_data/S*/s*.mat')
    files += glob.glob('/Users/nwilming/u/conf_data/S*/s*.mat')
    files += glob.glob('/Users/nwilming/u/conf_data/s*/s*.mat')
    files += glob.glob('/Users/nwilming/u/conf_data/S*/S*.mat')

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

def plot_kernel(y, response,  **kw):
    r = unique(response)
    del kw['label']
    sns.tsplot(vstack(y[response==r[0]].values), arange(0.5, 9.6, 1), **kw)
    sns.tsplot(vstack(y[response==r[1]].values), arange(0.5, 9.6, 1), **kw)
    #plot(arange(0.5, 9.6, 1), vstack(y[response==r[0]].values).mean(0), **kw)
    #plot(arange(0.5, 9.6, 1), vstack(y[response==r[1]].values).mean(0),   **kw)


def bootstrap(v, n, N, func=nanmean, alpha=.05):
    '''
    Bootstrap values in func(v[samples]), by drawing n samples N times.
    Returns CI whose width is specified by alpha.
    '''
    r = []
    for i in range(N):
        id_rs = np.random.randint(0, len(v), size = (n,))
        r.append(func(v[id_rs]))
    return prctile(r, [(alpha*100)/2, 50, 100 - (alpha*100)/2])

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
    low, high = prctile(xval, [1, 99])
    idx = (low<xval) & (xval<high)
    xval = xval[idx]
    data = data[idx]
    edges = prctile(xval, bins)
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
    results = log_res.fit()
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


def plot_model(df, model, bins=[linspace(0,.25,100), linspace(0,1,100)], hyperplane_only=False):
    ex, ey = bins
    M,C = meshgrid(*bins)
    resp1 = histogram2d(df[df.response==1].stdc.values, df[df.response==1].mc.values, bins=bins)[0] +1
    resp2 = histogram2d(df[df.response==-1].stdc.values, df[df.response==-1].mc.values, bins=bins)[0] +1
    resp1 = resp1.astype(float)/sum(resp1)
    resp2 = resp2.astype(float)/sum(resp2)

    p = model.predict(vstack([M.ravel(), C.ravel(), 0*ones(M.shape).ravel()]).T)
    p = p.reshape(M.shape)

    decision = lambda x: -(model.params.mc*x+ model.params.Intercept)/model.params.stdc
    if not hyperplane_only:
        pcolor(bins[1], bins[0], log(resp1/resp2))
    plot([0, 1], [decision(0.), decision(1.)], 'r', lw=2)
    plot([0.5, .5], [0, 1], 'r--', lw=2)
    ylim([0, 0.25])
    xlim([0, 1])
    #contour(bins[1], bins[0], p.T, [0.5])
    return bins, p
