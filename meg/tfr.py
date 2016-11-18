import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import artifacts, preprocessing
from conf_analysis.behavior import empirical, metadata, keymap
import glob
import mne, locale
import pandas as pd
import numpy as np
import cPickle


def tfr2df(tfr, freq, channel, tmin=None, tmax=None):
    '''
    Read out values for specific frequencies and channels from set of tfrs.

    channels is a list of channel names.
    freq is a list of frequencies
    '''
    freq = np.atleast_1d(freq)
    if tmin is not None and tmax is not None:
        tfr.crop(tmin, tmax)
    ch_ids = np.where(np.in1d(tfr.ch_names, channel))[0]

    ch_idx = np.in1d(np.arange(tfr.data.shape[1]), ch_ids)
    freq_idx = np.in1d(tfr.freqs, freq)
    tfr.data = tfr.data[:, ch_ids, :,:][:, :, np.where(freq_idx)[0], :]
    tfr.freqs = tfr.freqs[freq_idx]
    trials = np.arange(tfr.data.shape[0])

    
    trials, channel, freq, time = np.meshgrid(trials, ch_ids.ravel(),
                                              tfr.freqs.ravel(), tfr.times.ravel(),
                                              indexing='ij')
    print trials.shape, tfr.data.shape
    print ch_ids.shape, tfr.freqs.shape
    assert trials.shape==tfr.data.shape

    return pd.DataFrame({'trial':trials.ravel(), 'channel':channel.ravel(),
        'freq':freq.ravel(), 'time':time.ravel(), 'power':tfr.data.ravel()})
