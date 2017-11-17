import mne
import numpy as np

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing, localizer
from conf_analysis.meg import tfr_analysis as ta
from conf_analysis.meg import source_recon as sr
from joblib import Memory, Parallel, delayed
from mne import compute_covariance
from mne.beamformer import lcmv_epochs
from mne.time_frequency.tfr import _compute_tfr

import pandas as pd

from itertools import product, izip


memory = Memory(cachedir=metadata.cachedir)


def overview_figure(avg):
    '''
    Prepare data for an overview figure that shows source recon'ed activity.
    '''
    import pylab as plt
    # 1 Plot TFR for this participant and subject
    #ep = ta.get_sub_sess_object(subject, session, (10, 150), None, -0.4, 1.1)
    #avg = ep.average()
    #del ep
    #avg = avg.apply_baseline((-0.2, 0), mode='zscore')
    chan, f, t = peak_channel(avg, 20)
    plt.subplot(1, 2, 1)
    avg.plot_topomap(fmin=35, fmax=100, axes=plt.gca())
    plt.subplot(1, 2, 2)
    avg.plot([chan], axes=plt.gca())


def peak_channel(avg, fmin=10):
    id_f = fmin < avg.freqs
    chan, f, t = np.unravel_index(
        np.argmax(avg.data[:, id_f, :]),
        avg.data[:, id_f, :].shape)
    return chan, f+fmin, t


def make_averaged_sr(stcs, subject, session, lowest_freq, prefix=''):
    if lowest_freq is None:
        lowest_freq = 'None'
    stcs = reduce(lambda x, y: x + y, stcs) / len(stcs)
    filename = '/home/nwilming/conf_meg/source_recon/%sSR_S%i_SESS%i_F%s.stc' % (
        prefix, subject, session, str(lowest_freq))
    idbase = (-.5 < stcs.times) & (stcs.times < 0)
    m = stcs.data[:, idbase].mean(1)[:, np.newaxis]
    s = stcs.data[:, idbase].std(1)[:, np.newaxis]
    stcs.data = (stcs.data - m) / s
    stcs.save(filename)


@memory.cache
def get_cov(epochs):
    return compute_covariance(epochs, tmin=0, tmax=1, method='shrunk')


@memory.cache
def get_noise_cov(epochs):
    return compute_covariance(epochs, tmin=-0.5, tmax=0, method='shrunk')


def extract(subject, session, lowest_freq=None, run_epochs=True,
            run_localizer=False):
    epochs, meta = preprocessing.get_epochs_for_subject(subject,
                                                        'stimulus',
                                                        sessions=session)
    epochs = epochs.pick_channels(
        [x for x in epochs.ch_names if x.startswith('M')])
    epochs.times -= 0.75
    epochs._raw_times -= 0.75
    # epochs.crop(tmin=-.5, tmax=1.4) #.decimate(2)

    epochs = epochs.apply_baseline((-0.25, 0))
    if lowest_freq is not None:
        epochs.filter(lowest_freq, None)
    data_cov = get_cov(epochs)
    noise_cov = None  # get_noise_cov(epochs)
    forward, bem, source, trans = sr.get_leadfield(subject, session)
    labels = sr.get_labels(subject)
    labels = [l for l in labels if 'V' in l.name]
    # combined_label = reduce(lambda x, y: x + y, labels)
    source_epochs, source_stcs = None, None
    source_localizer, loc_stcs = None, None
    if run_epochs:
        source_epochs, source_stcs = do_epochs(epochs, meta, forward, source,
                                               noise_cov,
                                               data_cov,
                                               labels)

    if run_localizer:
        localizer_epochs = localizer.get_localizer(subject)
        localizer_epochs = localizer_epochs.pick_channels(
            [x for x in localizer_epochs.ch_names if x.startswith('M')])
        source_localizer, loc_stcs = do_epochs(
            localizer_epochs, None, forward, source, noise_cov,
            data_cov, labels)
    return source_epochs, source_stcs, source_localizer, loc_stcs


def do_epochs(epochs, meta, forward, source, noise_cov, data_cov, labels,
              save_stcs=True):
    results = []
    stcs = []
    if labels is None:
        labels = []
    if meta is None:
        index = np.arange(epochs._data.shape[0])
    else:
        index = meta.index.values
    for trial, epoch in izip(index,
                             lcmv_epochs(epochs, forward, noise_cov, data_cov,
                                         reg=0.05,
                                         pick_ori='max-power',
                                         return_generator=True)):
        srcepoch = {'time': epoch.times, 'trial': trial}

        for label in labels:
            pca = epoch.extract_label_time_course(
                label, source, mode='pca_flip')
            srcepoch[label.name] = pca
        results.append(srcepoch)
        if save_stcs:
            stcs.append(epoch)
        else:
            del epoch
    if len(labels) > 0:
        results = pd.concat([to_df(r) for r in results])
    else:
        results = None
    return results, stcs


def to_df(r):
    p = {}
    for key in r.keys():
        if key == 'trial':
            p[key] = np.array([r[key]] * len(r['time']))
        else:
            p[key] = r[key].ravel()
    return pd.DataFrame(p)


def source_tfr(epochs, foi=None, sfreq=600, cycles=None, time_bandwidth=None,
               decim=10, n_jobs=4, **kwargs):
    '''
    Epochs is a data frame with time and trial as index.
    '''
    tfr_params = dict(n_cycles=cycles, n_jobs=n_jobs, use_fft=True,
                      zero_mean=True, time_bandwidth=time_bandwidth)

    results = []
    # for trial, data in epochs.groupby('trial'):
    values = np.stack([data.T.values[:, :]
                       for trial, data in epochs.groupby('trial')])
    trials = np.array([trial
                       for trial, data in epochs.groupby('trial')])
    # Epochs needs to be a (n_epochs, n_channels, n_times) array.
    power = _compute_tfr(values, foi,
                         sfreq=sfreq,
                         method='multitaper',
                         decim=decim,
                         output='complex',
                         **tfr_params)
    power = 10 * np.log10(np.abs(power)**2)
    # power is a (n_epochs, n_channels, n_freqs, n_times) array.
    columns = data.columns
    index = data.index.get_level_values('time')[::decim]

    for (f_idx, freq), (e_idx, epoch) in product(enumerate(foi),
                                                 enumerate(trials)):
        df = pd.DataFrame(power[e_idx, :, f_idx, :].T,
                          index=index, columns=columns)
        df.loc[:, 'trial'] = epoch
        df.loc[:, 'freq'] = freq
        results.append(df)
    return pd.concat(results)


@memory.cache(ignore=['n_jobs'])
def get_tfr(df, foi, cycles, tb, n_jobs=12):
    gp = source_tfr(df,
                    foi=[foi], decim=2, cycles=[cycles],
                    time_bandwidth=tb, n_jobs=n_jobs, verbose='WARNING')
    del gp['freq']
    gp = gp.reset_index().set_index(['trial', 'time'])
    gp = gp.query('-0.5 < time & time < 1.4')
    columns = [x.replace('lh.wang2015atlas.', '')
                .replace('rh.wang2015atlas.', '')
               for x in gp.columns]
    gp.columns = columns
    col_sets = {}
    for hemi, ud in product(['lh', 'rh'], ['d', 'v']):
        cols = [x for x in gp.columns if (hemi in x) and (ud in x)]
        col_sets[hemi + ud] = gp.loc[:, cols].mean(1)
    gp = pd.DataFrame(col_sets)
    return gp


def zscore(o):
    '''
    Convert to percent change
    o is a pivoted df with time in columns and epochs in trials
    '''

    base = o.loc[:,  slice(-0.5, 0)].values.mean()
    std = o.loc[:, slice(-0.5, 0)].values.std()
    return (o - base) / std


def sample_aligned(time_locked, sample, area, n=181):
    times = time_locked.columns.values
    t_start = sample * 0.1 - 0.1
    dt = np.diff(times).mean()

    t_end = 0.35
    n_step = int(np.ceil((t_end + 0.1) / dt))
    n_start = np.argmin(abs(times - t_start))

    df = time_locked.iloc[:, n_start:n_start + n_step]
    df.columns = df.columns.values - t_start
    return df


def total_power(df, area):
    v1 = pd.pivot_table(df, index='trial', columns='time', values=area)
    v2 = zscore(v1)
    tp = v2.loc[:, slice(0.15, 1.15)].mean(1)
    return np.trapz(tp.values, tp.index.values)


def contrast_effect(df, meta, area):
    time_locked = pd.pivot_table(
        df.reset_index(), index='trial', columns='time', values=area)
    time_locked = time_locked - time_locked.mean(0)
    time_locked = zscore(time_locked)
    cvals = meta.loc[time_locked.index, 'contrast_probe']
    cvals = np.stack(cvals)
    lows, intms, highs = [], [], []

    for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        high = cvals[:, s] > 0.6
        intm = (0.4 < cvals[:, s]) & (cvals[:, s] < 0.6)
        low = cvals[:, s] < 0.4

        k = sample_aligned(time_locked, s, area)
        base = k.loc[:, -0.1:0.05].mean(1)
        ps = (k.values - base[:, np.newaxis])
        lows.append(ps[low, :].mean(0))
        highs.append(ps[high, :].mean(0))
        intms.append(ps[intm, :].mean(0))
    lows, highs, intms = np.stack(lows), np.stack(highs), np.stack(intms)
    tp = np.trapz(highs.mean(0) - lows.mean(0), k.columns.values)
    return lows, intms, highs, tp


def get_params(df, meta, ps, area):
    gp = get_tfr(df, *ps, n_jobs=12)
    tp_evoked = total_power(gp, area)
    lows, intms, highs, tp = contrast_effect(gp, meta, area)
    return ps[0], tp_evoked, tp


def get_lcmv_source(subject):
    meta = preprocessing.get_meta_for_subject(subject, 'stimulus')
    df = pd.read_hdf('/home/nwilming/conf_meg/S%i-lcmv.hdf' %
                     subject, 'epochs')
    df = df.set_index(['time', 'trial'])
    columns = [x.replace('lh.wang2015atlas.', '').replace(
        'rh.wang2015atlas.', '') for x in df.columns]
    df.columns = columns
    return df, meta


@memory.cache
def get_subs(subject, params, area):
    meta = preprocessing.get_meta_for_subject(subject, 'stimulus')
    df = pd.read_hdf('/home/nwilming/conf_meg/S%i-lcmv.hdf' %
                     subject, 'epochs')
    df = df.set_index(['time', 'trial'])
    columns = [x.replace('lh.wang2015atlas.', '').replace(
        'rh.wang2015atlas.', '') for x in df.columns]
    df.columns = columns
    tps = Parallel(n_jobs=1)((delayed(get_params)(df, meta, ps, area)
                              for ps in params))
    tps = np.array(tps)
    tps = pd.DataFrame({'effect': tps[:, 2], 'freq': tps[:, 0]})
    tps.loc[:, 'subject'] = subject
    return tps
