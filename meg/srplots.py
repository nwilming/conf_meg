import numpy as np
import pylab as plt
import pandas as pd
from conf_analysis.meg import preprocessing
from conf_analysis.behavior import metadata
import seaborn as sns
from joblib import Parallel, delayed
# from scipy.ndimage import gaussian_filter
from joblib import Memory
from glob import glob
# from sklearn import neighbors
# from conf_analysis.meg import tfr_analysis as ta
import re


memory = Memory(cachedir=metadata.cachedir)


def get_freq_tuning(subject, tuning=1):
    power_files = []
    stc_files = []
    frequencies = {}
    for sess in range(4):
        fs = glob('/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-F*-lcmv.hdf' %
                  (subject, sess))

        def key(x):
            y = parse(x, ['F'])['F'][0]
            return int(y)

        fs.sort(key=key)
        power_files += [fs[tuning]]
        frequencies[sess] = key(fs[tuning])
        sstring = ('/home/nwilming/conf_meg/source_recon/SR_S%i_SESS%i_lF*_F%i*.stc' %
                   (subject, sess, frequencies[sess]))
        s = glob(sstring)
        stc_files.append(s)
    return frequencies, power_files, stc_files


def get_power(subject, session=None, decim=3, tuning=None, F=None):
    '''
    Tuning determines which frequency to load
    '''
    if (tuning is None) and (F is None):
        raise RuntimeError(
            'Either tuning or F must be set, but both are None.')
    if tuning is not None:
        freqs, files, _ = get_freq_tuning(subject, tuning=tuning)
        if session is not None:
            files = files[session]
        print files
    else:
        files = []
        for sess in range(4):
            sstring = ('/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-F%i*-lcmv.hdf' %
                       (subject, sess, F))
            files.extend(glob(sstring))
    if len(files) == 0:
        raise RuntimeError(
            'No files found for sub %i, session %i' % (subject, session))
    df = []
    for f in preprocessing.ensure_iter(files):
        d = pd.read_hdf(f)
        if d.time.min() < -0.75:
            d.time += 0.75
        if decim is not None:
            def decim_func(df, x):
                df = df.reset_index().set_index('time')
                df = df.sort_index()
                return df.iloc[slice(0, len(df), x)]
            d = d.groupby('trial').apply(lambda x: decim_func(x, decim))
            del d['trial']
        d = d.reset_index().set_index(['time', 'trial'])
        d = combine_areas(d, hemi=True)
        df.append(d)
    df = pd.concat(df)

    meta = preprocessing.get_meta_for_subject(
        subject, 'stimulus', sessions=session)
    return df, meta


def sample_aligned_power(df, meta, area, baseline=(-0.2, 0)):
    darea = pd.pivot_table(df.reset_index(), values=area,
                           index='trial', columns='time')
    # darea = np.log10(darea)
    bmin, bmax = baseline
    darea = darea.subtract(darea.loc[:, bmin:bmax].mean(1), 'index')
    darea = darea.div(darea.loc[:, bmin:bmax].std(1), 'index')
    darea = darea.subtract(darea.mean(0))

    cvals = np.vstack(meta.loc[darea.index.values, 'contrast_probe'])
    stuff = []

    for i, sample in enumerate(cvals.T):
        power = darea.loc[:, (i * 0.1) - 0.1:i * 0.1 + 0.4]
        base = darea.loc[:, (i * 0.1) - 0.1: (i * 0.1)].mean(1)
        power = power.subtract(base, axis=0)
        #print (i * 0.1) - 0.1, base.head()
        time = power.columns.values
        power.loc[:, 'contrast'] = sample
        power.loc[:, 'sample'] = i
        power = power.reset_index().set_index(['trial', 'contrast', 'sample'])
        power.columns = np.around(time - time.min() - 0.1, 3)
        power.columns.name = 'time'
        power = power.stack().reset_index()
        power.columns = ['trial', 'contrast', 'sample', 'time', area]
        stuff.append(power)
    return pd.concat(stuff)


def plot_sample_aligned_power(stuff, area, edges=[0, 0.5, 1], ax=None):
    k = stuff.groupby(['time', pd.cut(stuff.contrast, edges)]).mean()
    del k['contrast']
    pd.pivot_table(k, index='contrast', columns='time',
                   values=area).T.plot(ax=ax)


def plot_sample_aligned_power_all_areas(df, meta, edges, plot_areas=['V1', 'V2', 'V3', 'V4']):
    import matplotlib
    gs = matplotlib.gridspec.GridSpec(4, 4)
    areas = df.columns
    cnt = 0
    plot_pos = {'V1vlh': (0, 0), 'V1vrh': (0, 1), 'V1dlh': (1, 0), 'V1drh': (1, 1),
                'V2vlh': (2, 0), 'V2vrh': (2, 1), 'V2dlh': (3, 0), 'V2drh': (3, 1),
                'V3vlh': (0, 2), 'V3vrh': (0, 3), 'V3dlh': (1, 2), 'V3drh': (1, 3),
                'hV4lh': (2, 2), 'hV4rh': (2, 3)}
    for area, pos in plot_pos.iteritems():
        if not any([a in area for a in plot_areas]):
            continue
        plt.subplot(gs[pos[0], pos[1]])
        s = sample_aligned_power(df, meta, area)
        plot_sample_aligned_power(
            s, area, edges, ax=plt.gca())
        #plt.title(area)
        plt.axvline(0, color='k', alpha=0.5)
        plt.ylim([-1.5, 1.5])        
        plt.legend([])
        #plt.text(0, 7.5, area)
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    import seaborn as sns
    sns.despine()
    return gs


def combine_areas(df, hemi=False):
    areas = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d',
             'hV4', 'VO2', 'PHC1', 'PHC2',
             'TO1', 'LO1', 'LO2', 'V3A', 'IPS0', 'IPS1', 'IPS3',
             'IPS4', 'IPS5', 'FEF']
    res = []

    def foo(df, areas, hemi=''):
        res = []
        for area in areas:
            cols = [c for c in df.columns if (area in c)]
            name = area + hemi
            if len(cols) >= 1:
                col = df.loc[:, cols].mean(1)
                col.name = name
                res.append(col)
        return res
    if hemi:
        res.extend(
            foo(df.loc[:, [x for x in df.columns if 'rh' in x]], areas, hemi='rh'))
        res.extend(
            foo(df.loc[:, [x for x in df.columns if 'lh' in x]], areas, hemi='lh'))
    else:
        res.extend(foo(df, areas))

    return pd.concat(res, axis=1)


def get_dfp_for_all_subs():
    frames = []
    for subject in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]:
        df, meta = get_power(subject, 59)
        dfp = combine_areas(df)
        areas = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d',
                 'hV4', 'VO2', 'PHC1', 'PHC2',
                 'TO1', 'LO1', 'LO2', 'V3A', 'IPS0', 'IPS1', 'IPS3',
                 'IPS4', 'IPS5', 'FEF']
        for i, area in enumerate(areas):
            saligned = sample_aligned_power(dfp, meta, area)
            saligned.loc[:, 'subject'] = subject
            saligned.loc[:, 'area'] = area
            frames.append(saligned)
    return frames


def fir_analysis(df, meta, area='V1dlh',
                 edges=[-0.1, .3, 0.5, 0.7, 1.1], Hz=300):
    trials = pd.pivot_table(
        df, values=area, index='trial', columns='time')
    cvals = np.stack(meta.loc[trials.index.values, :].contrast_probe)
    mm = trials.loc[:, -0.2:0].values.mean()
    ss = trials.loc[:, -0.2:0].values.std()
    trials = (trials.values - mm) / ss
    trials = trials - trials.mean(0)[np.newaxis, :]

    events = np.digitize(cvals, edges)

    event_times = np.stack(
        [np.arange(0, 1, 0.1) + 0.75 + i * 2.25
         for i in range(cvals.shape[0])])

    events = [event_times[events == i].ravel()
              for i in np.unique(events)]
    # return event_times, trials.ravel()
    from fir import FIRDeconvolution
    fd = FIRDeconvolution(
        signal=trials.ravel()[np.newaxis, :],
        events=events,
        # event_names=['event_1'],
        sample_frequency=Hz,
        deconvolution_frequency=Hz,
        deconvolution_interval=[-0.1, 0.3])
    fd.create_design_matrix()
    fd.regress()
    return fd, trials


def parse(string, tokens):
    '''
    Extract all numbers following token.
    '''
    numbers = dict((t, [int(n.replace(t, ''))
                        for n in re.findall(t + '\d+', string)])
                   for t in tokens)
    return numbers


def get_sample_traces(df, area, meta, sample, edges, baseline=(-0.25, 0)):
    '''
    Return power traces grouped by contrast (defined by edges)
    '''
    ko = df.loc[:, area]
    ko = ko.unstack('time').loc[:, -0.25:1.4]
    ko.columns = ko.columns
    # ko = ko - ko.mean()
    cvals = pd.Series(np.vstack(meta.loc[ko.index.values, 'contrast_probe'])[
                      :, sample], index=ko.index)
    cvals.name = 'S3c'
    ko.index = cvals
    values = ko.groupby(pd.cut(ko.index, edges)).mean()
    base = values.loc[:, slice(*baseline)].mean(1)
    values = values.subtract(base, axis=0)
    return values


def plot_sample_traces(df, meta, area):
    from scipy.ndimage import gaussian_filter

    edges = [0] + list(np.linspace(0.3, 0.7, 3)) + [1]
    plt.figure(figsize=(12, 6))
    for sample in range(10):
        plt.subplot(2, 5, sample + 1)
        s = get_sample_traces(df, area, meta, sample, edges)
        for i, row in enumerate(s.values):
            plt.plot(s.columns.values, row, label=s.index.values[i])

        plt.axvline(sample * 0.1, color='k')
    plt.legend()
    sns.despine()
    plt.tight_layout()


def get_correlations(df, meta, area):
    '''
    Compute correlations between power and contrast per time point and
    contrast sample
    '''
    stuff = []
    darea = pd.pivot_table(df.copy(), values=area,
                           index='trial', columns='time')
    darea = darea.subtract(darea.loc[:, 0.5:0.75].mean(1), 'index')
    cvals = np.vstack(meta.loc[darea.index.values, 'contrast_probe'])
    for cp in range(10):
        res = {}
        cc = [np.corrcoef(cvals[:, cp], darea.loc[:, col].values)[0, 1]
              for col in darea]
        res['area'] = area
        res['sample'] = cp
        res['corr'] = cc
        res['time'] = darea.columns.values
        stuff.append(pd.DataFrame(res))
    return pd.concat(stuff)


def get_all_correlations(df, meta, n_jobs=12):
    generator = (delayed(get_correlations)(
        df.loc[:, area].reset_index(),
        meta,
        area)
        for area in df.columns)
    cc = Parallel(n_jobs=n_jobs)(generator)
    return pd.concat(cc)


def plot_area(cc, **kwargs):
    vals = []
    for d, c in cc.groupby(['sample']):
        time = c.time.values - 0.75
        idx = (0.1 * d <= time) & (time < (0.1 * d + 0.4))
        vals.append(c.loc[idx, 'corr'])
        plt.plot(time[idx] - (0.1 * d),  # (0.1 * d) +
                 c.loc[idx, 'corr'], **kwargs)
        plt.xlim([-0.25, 0.5])
        # yl = [(0.1 * d) - 0.1, (0.1 * d) + 0.1]
        plt.plot([0, 0], (-0.2, 0.2), color='k', alpha=0.5)
    vals = np.stack(vals).mean(0)
    plt.plot(time[idx] - (0.1 * d), vals, lw=3.5, color='k', alpha=0.75)
