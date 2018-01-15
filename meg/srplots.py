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
        fs = glob('/home/nwilming/conf_meg/sr_labeled/S%i-SESS%i-F*-tune%i-lcmv.hdf' %
                  (subject, sess, tuning))

        def key(x):
            y = parse(x, ['F'])['F'][0]
            return int(y)

        fs.sort(key=key)
        power_files += fs
        frequencies[sess] = key(fs[0])
        sstring = ('/home/nwilming/conf_meg/source_recon/SR_S%i_SESS%i_lF*_F%i*.stc' %
                   (subject, sess, frequencies[sess]))
        s = glob(sstring)
        stc_files.append(s)
    return frequencies, power_files, stc_files


def get_power(subject, decim=3):
    '''
    Tuning determines which frequency to load
    '''
    sstring = ('/home/nwilming/conf_meg/sr_labeled/S%i-SESS*-lcmv.hdf'
               % (subject))
    files = glob(sstring)
    if len(files) == 0:
        raise RuntimeError(
            'No files found for sub %i, session %i' % (subject))
    df = []
    for f in preprocessing.ensure_iter(files):
        subject, session, F, tuning = find_numbers(f)
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
        d.loc[:, 'subject'] = subject
        d.loc[:, 'session'] = session
        d.loc[:, 'F'] = F
        d.loc[:, 'tune'] = tuning
        d = d.reset_index().set_index(
            ['time', 'trial', 'subject', 'session', 'F', 'tune'])
        d = combine_areas(d, hemi=True)
        df.append(d)
    df = pd.concat(df)

    meta = preprocessing.get_meta_for_subject(
        subject, 'stimulus')
    return df, meta


def sample_aligned_power_AAA(df, meta, baseline=(-0.2, 0)):
    cols = []
    for area in df.columns:
        col = sample_aligned_power(df, meta, area, baseline=baseline)
        cols.append(col.set_index(['trial', 'contrast', 'sample', 'time']))
    return pd.concat(cols, axis=1)


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


def compute_tuning_index(sa, edges=[0, 0.4, 0.6, 1.]):
    areas = sa.columns
    sa = sa.reset_index()

    k = sa.groupby(['time', pd.cut(sa.loc[:, 'contrast'], edges)]).mean()
    del k['contrast']
    del k['sample']
    del k['trial']
    res = {}
    for area in areas:
        kk = pd.pivot_table(k, index='contrast', columns='time',
                            values=area).loc[:, 0:0.3]
        low = kk.iloc[0, :]
        high = kk.iloc[2, :]
        time = low.index.values
        res[area] = np.trapz(high.values, time) - np.trapz(low.values, time)
    return pd.Series(res)


def plot_sample_aligned_power(stuff, area, edges=[0, 0.5, 1], ax=None):
    k = stuff.groupby(['time', pd.cut(stuff.contrast, edges)]).mean()
    del k['contrast']
    k = pd.pivot_table(k, index='contrast', columns='time',
                       values=area)
    plt.plot(k.columns.values, k.values.T)


def plot_sample_aligned_power_all_areas(df, meta, edges, gs=None, plot_areas=['V1', 'V2', 'V3', 'V4']):
    if gs is None:
        import matplotlib
        gs = matplotlib.gridspec.GridSpec(4, 2)
    areas = df.columns
    cnt = 0
    plot_pos = {'V1-lh': (0, 0), 'V1-rh': (0, 1), 'V2-lh': (1, 0), 'V2-rh': (1, 1),
                'V3-lh': (2, 0), 'V3-rh': (2, 1), 'hV4-lh': (3, 0), 'hV4-rh': (3, 1)}
    for area, pos in plot_pos.iteritems():
        if not any([a in area for a in plot_areas]):
            continue
        plt.subplot(gs[pos[0], pos[1]])
        s = sample_aligned_power(df, meta, area)
        plot_sample_aligned_power(
            s, area, edges, ax=plt.gca())
        # plt.title(area)
        plt.axvline(0, color='k', alpha=0.5)
        plt.ylim([-1.5, 1.5])
        plt.legend([])
        plt.title(area)
        #plt.text(0, 7.5, area)
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    import seaborn as sns
    sns.despine()
    return gs


def combine_areas(df, only_lower=True, hemi=True):
    import re
    areas = ['V1', 'V2',  'V3', 'hV4']
    if not only_lower:
        areas += ['VO2', 'PHC1', 'PHC2',
                  'TO1', 'LO1', 'LO2', 'IPS0', 'IPS1', 'IPS3',
                  'IPS4', 'IPS5', 'FEF']
    res = []
    if hemi:
        areas = [a + '.?-lh' for a in areas] + [a + '.?-rh' for a in areas]
    for area in areas:

        index = [True if re.search(
            area, x) is not None else False for x in df.columns]

        col = df.loc[:, index].mean(1)
        col.name = area.replace('.?', '') if hemi else area
        res.append(col)
    return pd.concat(res, axis=1)


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


def find_numbers(string, ints=False):
    # From Marc Maxson on stackoverflow:
    numexp = re.compile(r'[-]?\d[\d,]*[\.]?[\d{2}]*')  # optional - in front
    numbers = numexp.findall(string)
    numbers = [x.replace(',', '') for x in numbers]
    if ints is True:
        return [int(x.replace(',', '').split('.')[0]) for x in numbers]
    else:
        return [float(x) for x in numbers]
