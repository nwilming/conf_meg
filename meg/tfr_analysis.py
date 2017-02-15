import sys
sys.path.append('/home/nwilming/')
import glob
from conf_analysis.behavior import metadata
import mne
import pandas as pd
import numpy as np
import cPickle
import pylab as plt
import seaborn as sns
from pymeg import tfr
sns.set_style('ticks')
from joblib import Memory
memory = Memory(cachedir=metadata.cachedir)


def plot_contrast(df):
    '''
    Make a plot that shows the dependence of gamma and contrast for a specific contrast
    sample over time. In detail: plot contrast vs gamma.
    '''
    pass


def contrast_dependence(df, contrast, edges=np.linspace(0.1, 0.9, 15)):
    '''
    Compute contrast vs. power in a freq band.

    df is a data frame with time in columns and indexed by trial, channel and freq.
    Data in df is assumed to be baseline corrected.
    '''
    # Bring data frame into trial x time form.
    df = df.groupby(level='trial').mean()
    #df = df.join(contrast)
    centers = [low+(high-low)/2. for low, high in zip(edges[:-1], edges[1:])]
    #print df.columns
    return df.groupby(pd.cut(contrast, bins=edges, labels=centers)).mean()

def get_all_tfr():
    pass


def get_subject(snum, freq, channel, tmin, tmax,
                epoch='stimulus'):
    filenames = glob.glob('/home/nwilming/conf_meg/S%i/*%stfr.hdf5'%(snum, epoch))
    metafiles = glob.glob('/home/nwilming/conf_meg/S%i/*%s.meta'%(snum, epoch))
    avg = tfr.get_tfrs(filenames, freq=freq, channel=channel, tmin=tmin, tmax=tmax)
    avg.loc[:, 'snum'] = snum
    avg.set_index('snum', append=True, inplace=True)
    avg.sort_index(inplace=True)
    meta = pd.concat([pd.read_hdf(f) for f in metafiles])
    return avg, meta


def baseline(avg, id_base):
    '''
    Baseline correction by dividing by average baseline
    '''
    base = avg.loc[:, id_base].values.ravel().mean()
    avg = np.log10(avg / base)
    return avg


def avg_baseline(avg, baseline):
    '''
    Baseline correction by dividing by average baseline
    '''
    time =  avg.columns.get_level_values('time').values.astype(float)
    id_base = (baseline[0]<time) & (time<baseline[1])
    avg = avg.apply(np.log10)
    # Baseline correction with average baseline across all trials. (base is freq specific)
    base = avg.loc[:, id_base].groupby(level='freq').mean().mean(1) # This should be len(#Freq)
    def div(x):
        bval = base.loc[x.index.get_level_values('freq').values[0]]
        return x - bval
    return avg.groupby(level='freq').apply(div)


def trial_baseline(avg, baseline):
    '''
    Baseline correction by dividing by per-trial baseline
    '''
    index = avg.index
    time =  avg.columns.get_level_values('time').values.astype(float)
    id_base = (baseline[0]<time) & (time<baseline[1])
    avg = np.log10(avg)
    base = avg.loc[:, id_base].groupby(level=['freq', 'trial']).mean().mean(1)
    levels = list(set(avg.index.names) - set(baseline.index.names))
    avg.index = avg.index.droplevel(levels)
    avg.div(base, axis=0, inplace=True)
    avg.index = index
    return avg


def avg_contrast_resp(avg, meta, edges=np.linspace(0.2, 0.8, 7),
                      cidx=slice(0, 10),
                      correct_quest=False):
    '''
    avg needs to be indexed by freq and trial hash
    '''
    n_groups = len(edges)-1
    cv = np.vstack(meta.contrast_probe)
    if correct_quest:
        cv = cv-(meta.contrast*meta.side)[:, np.newaxis]
    cv = cv[:, cidx].mean(1)

    contrast = pd.Series(cv, index=
                         meta.index.get_level_values('hash'))
    grouper = (pd.cut(contrast, edges, labels=np.arange(n_groups))
                        .reset_index())
    crs = []

    for i, (c, o) in enumerate(grouper.groupby(0)):
        m = (avg.loc[(list(o.hash.values),), :]
             .groupby(level='freq')
             .mean())
        if len(m)==0:

            continue
        m.loc[:, 'cgroup'] = c
        m.set_index('cgroup', append=True, inplace=True)
        crs.append(m)
    return pd.concat(crs)


def plot_by_freq(resp, **kwargs):
    if 'vmin' not in kwargs.keys():
        kwargs['vmin'] = -0.5
    if 'vmax' not in kwargs.keys():
        kwargs['vmax'] = 0.5
    if 'cmap' not in kwargs.keys():
        kwargs['cmap'] = 'RdBu_r'

    n_groups = len(np.unique(resp.index.get_level_values('cgroup')))
    for i, (c, m) in enumerate(resp.groupby(level='cgroup')):
        time = np.unique(m.columns.get_level_values('time'))
        tc = np.array([time[0]] + list([(low+(high-low)/2.)
            for low, high in zip(time[:-1], time[1:])]) + [time[-1]])
        freqs = np.unique(m.index.get_level_values('freq'))
        fc = np.array([freqs[0]] + list([(low+(high-low)/2.)
            for low, high in zip(freqs[:-1], freqs[1:])]) + [freqs[-1]])

        plt.subplot(1, n_groups, i+1)
        plt.pcolormesh(tc, fc,
                m.values, **kwargs)
        #plt.xlim([-0.4, 1.1])
        plt.ylim([min(freqs), max(freqs)])
        if i>0:
            plt.yticks([])
        else:
            plt.ylabel('Frequency')
            plt.xlabel('Time')
        plt.xticks([-.4, 0, .4, .8])


def plot_avg_freq(resp, freqs=slice(60, 90)):
    n_groups = len(np.unique(resp.index.get_level_values('cgroup')))
    colors = sns.color_palette("coolwarm", n_groups)

    for i, (c, m) in enumerate(resp.groupby(level='cgroup')):
        time = m.columns.get_level_values('time')
        m = m.loc[(freqs,), :]
        y = m.values.mean(0)
        plt.plot(time, y, label=c, color=colors[i])
        plt.xlabel('Time')
        plt.xticks([-.4, 0, .4, .8])


@memory.cache
def get_gamma_specific_data(snum, df):
    loc = cPickle.load(open('localizer_results/lr_%i_gamma.pickle'%snum))
    id_cluster = np.argmin([len(loc[0]['ch_names']), len(loc[1]['ch_names'])])
    channels= loc[id_cluster]['ch_names']
    froi = loc[id_cluster]['froi']
    freqs = [froi+df[0], froi+df[1]]
    avg, meta = get_subject(snum, freqs, channels, tmin=-0.4, tmax=1.1)
    return avg, meta


def remove_condition_means(avg, meta):
    '''
    Remove mean response to side condition.
    '''
    fluctuations=[]
    metas = []
    for condition, cd in meta.groupby('side'):
        index = list(cd.index.values)
        cond_data = avg.loc[(index,),:]
        fl = cond_data.groupby(level=['freq', 'channel']).apply(lambda x: x-x.mean())
        fluctuations.append(fl)
        metas.append(cd)
    return pd.concat(fluctuations), pd.concat(metas)


def response_specific_cr(avg, meta, edges=6, dt=slice(0,10)):
    cs = []
    for mr, m in meta.groupby('side'):
        index = list(m.index.values)
        cvals = np.vstack(m.contrast_probe)#-(m.contrast*m.side)[:, np.newaxis]
        mc = np.mean(cvals.ravel())
        t = np.min(np.abs(np.percentile(cvals.ravel(), [1, 99])-0.5))
        spec_edges = np.linspace(mc-t, mc+t, edges)
        print mc, np.mean(m.contrast), spec_edges
        c = contrast_response(avg.loc[(index,),:], m, spec_edges, dt=dt)
        c.loc[:, 'side'] = mr
        c.set_index('side', append=True, inplace=True)
        cs.append(c)
    return pd.concat(cs)


def contrast_response(avg, meta, edges=np.linspace(0.3, 0.7, 5), dt=slice(0, 10)):
    try:
        cs = []
        for t in dt:
            c = avg_contrast_resp(avg, meta,
                edges=edges, cidx=t)
            c.loc[:, 'dt'] = t.start
            c.set_index('dt', append=True, inplace=True)
            cs.append(c)
        return pd.concat(cs)
    except TypeError:
        return avg_contrast_resp(avg, meta,
            edges=edges, cidx=dt)


def fit_sigmoid(x, y):
    from scipy.optimize import curve_fit
    from scipy.stats import linregress
    from conf_analysis.behavior.empirical import asfuncof
    xr, yr = asfuncof(x, y, np.linspace(1, 99, 25))
    xr, yr = np.array(xr), np.array(yr)
    def sigmoid(x, A, K, B, v, Q):
        y = A+((K-A)/(1+Q*np.e**(-B*x))**(1./v))
        return y

    try:
        popt, pcov = curve_fit(sigmoid, x, y)
        rss_sigmoid = sum((sigmoid(xr, *popt)-yr)**2)
    except RuntimeError:
        popt, rss_sigmoid = [np.nan]*5, np.nan

    plin  = linregress(x, y)
    rss_lin = sum(((plin[0]*xr+plin[1])-yr)**2)
    return popt, rss_sigmoid, plin, rss_lin


def correlation_analysis(snum, df=(-10, 30)):
    avg, meta = get_gamma_specific_data(snum, df)
    avg = avg_baseline(avg, (-0.4, 0))
    avg = avg.groupby(level=['trial', 'snum']).mean()
    index = avg.index.get_level_values('trial')
    meta = meta.loc[index, :]
    cvals = np.vstack(meta.contrast_probe)

    def closest(x, a):
        '''
        Return column of a closest in time to x.
        '''
        id_t = np.argmin(abs(a.columns.values.astype(float)-x))
        return a.iloc[:, id_t]

    times = avg.columns.values.astype(float)
    df = []
    for t in times:
        y = closest(t, avg)
        for stim_time in range(10):
            results = {}
            x = cvals[:, stim_time]
            results['correlation'] = np.corrcoef(x, y)[0,1]
            popt, rss_sig, plin, rss_lin = fit_sigmoid(x, y)
            results['p_sig'] = popt
            results['rss_sig'] = rss_sig
            results['p_lin'] = plin
            results['rss_lin'] = rss_lin
            results['time'] = t
            results['sample'] = stim_time
            df.append(results)
    return df

def do_rm_anova(responses):
    groups = [cd.values for cg, cd in responses.groupby(level='cgroup')]
    p = np.dstack(groups)
    p=p.swapaxes(1,2)
    fs, ps = mne.stats.f_mway_rm(p, [p.shape[1]], 'A')
    return fs, ps


def draw_significance(ps, times, y=0, dy=0.1, alphas=[0.05, 0.01, 0.001],
    func=mne.stats.fdr_correction, **kwargs):
    #assert(len(np.unique(np.diff(np.around(times,4))))==1)
    for alpha in alphas:
        hs, ps = func(ps, alpha)
        dt = np.diff(times)[0]
        if sum(hs)>0:
            for l in np.where(hs)[0]:
                plt.plot([times[l]-dt/2., times[l]+dt/2.], [y, y], **kwargs)
            y+=dy
