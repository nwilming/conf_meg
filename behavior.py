import pandas as pd
from pylab import *
import seaborn as sns
sns.set_style('ticks')
import glob
from scipy.io import loadmat
import statsmodels.api as sm
from sklearn import linear_model
import patsy

def load_data():
    '''
    '''
    files = glob.glob('/Users/nwilming/u/conf_data/s*/s*.mat')
    files += glob.glob('/Users/nwilming/u/conf_data/s*/s*.csv')
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
        dfs.append(df)
    return pd.concat(dfs).reset_index()


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

def conf_kernels(df):
    legend_labels = {(-1., 2.): 'Yes, the 1st', (-1., 1.): 'Maybe, the 1st',
                     ( 1., 2.): 'Yes, the 2nd', (1., 1.): 'Maybe, the 2nd'}
    colors = dict((k, c) for k,c in zip(((-1., 2.), (-1., 1.), ( 1., 2.), (1., 1.)), sns.color_palette(n_colors=4)))
    for cond, group in df.groupby(['response', 'confidence']):
            plot(arange(0.5, 9.6, 1), vstack(group['contrast_probe'].values).mean(0), label=legend_labels[cond], color=colors[cond])
    ylim([0.35, .65])
    yticks([0.35, 0.5, 0.65])
    xlabel('time')
    sns.despine()
    #legend()



def fit_logistic(df, formula):
    y,X = patsy.dmatrices(formula, df, return_type='dataframe')
    print y.shape

    log_res = sm.GLM(y, X, family=sm.families.Binomial())
    results = log_res.fit()
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


def plot_model(df, model, bins=[linspace(0,.25,100), linspace(0,1,100)]):
    ex, ey = bins
    M,C = meshgrid(*bins)
    resp1 = histogram2d(df[df.response==1].stdc.values, df[df.response==1].mc.values, bins=bins)[0] +1
    resp2 = histogram2d(df[df.response==-1].stdc.values, df[df.response==-1].mc.values, bins=bins)[0] +1
    resp1 = resp1.astype(float)/sum(resp1)
    resp2 = resp2.astype(float)/sum(resp2)

    p = model.predict(vstack([M.ravel(), C.ravel(), 0*ones(M.shape).ravel()]).T).reshape(M.shape)
    decision = lambda x: -(model.params.mc*x+ model.params.Intercept)/model.params.stdc
    pcolor(bins[1], bins[0], log(resp1/resp2))
    plot([0, 1], [decision(0.), decision(1.)], 'r', lw=2)
    plot([0.5, .5], [0, 1], 'r--', lw=2)
    ylim([0, 0.25])
    xlim([0, 1])
    #contour(bins[1], bins[0], p.T, [0.5])
    return bins, p
