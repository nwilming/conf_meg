import mne
import numpy as np

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing, localizer
from conf_analysis.meg import source_recon as sr
from joblib import Memory
from mne import compute_covariance
from mne.beamformer import lcmv_epochs
from mne.time_frequency.tfr import _compute_tfr

import pandas as pd

from itertools import product, izip


memory = Memory(cachedir=metadata.cachedir)


@memory.cache
def get_cov(epochs):
    return compute_covariance(epochs, tmin=0, tmax=1, method='shrunk')


@memory.cache
def get_noise_cov(epochs):
    return compute_covariance(epochs, tmin=-0.5, tmax=0, method='shrunk')


def extract(subject, localizer_only=False):
    epochs, meta = preprocessing.get_epochs_for_subject(subject, 'stimulus')
    epochs.times -= 0.75
    epochs = epochs.apply_baseline((-0.25, 0))
    data_cov = get_cov(epochs)
    noise_cov = get_noise_cov(epochs)
    forward, bem, source, trans = sr.get_leadfield(subject)
    labels = sr.get_labels(subject)
    labels = [l for l in labels if 'V' in l.name]
    # combined_label = reduce(lambda x, y: x + y, labels)
    if not localizer_only:
        source_epochs = do_epochs(epochs, forward, source, noise_cov, data_cov, labels)
    else:
        source_epochs = None

    localizer_epochs = localizer.get_localizer(subject)
    localizer_epochs = localizer_epochs.pick_channels(
        [x for x in localizer_epochs.ch_names if x.startswith('M')])

    source_localizer = do_epochs(
        localizer_epochs, None, forward, source, noise_cov, data_cov, labels)
    return source_epochs, source_localizer


def do_epochs(epochs, meta, forward, source, noise_cov, data_cov, labels):
    results = []
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
        del epoch
    return pd.concat([to_df(r) for r in results])


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
