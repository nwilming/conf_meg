'''
Helper function for an example topo plot
'''
from mne import create_info
from mne.io import RawArray
import scipy.io as sio
import numpy as np
import h5py
import pandas as pd
from copy import copy

def events_to_samples(events, time, eid=None):
    '''
    Converts events from original 1200Hz space to downsampled 500Hz time series.
    The downsampled time-series is already cut into pieces, which makes this a bit
    more difficult.
    '''
    events = copy(events)*5./12
    time = copy(time)*500
    if eid is None:
        value = events*0
    return np.array([(np.argmin(abs(time-event)), 0, v) for (event, v) in zip(events, eid)]).T

def nan_artifacts(raw, artifacts):
    '''
    Fill artifact periods with NANs
    '''
    for group, defs in artifacts.iteritems():
        for start, end in defs:
            raw[:,start:end] = np.nan

def read_artifacts(fname, var_name='artifacts', scale=1.):
    # load Matlab/Fieldtrip data
    mat = h5py.File(fname)
    ft_data = mat[var_name]
    arts = {}
    for group in ft_data.keys():
        arts[group] = np.array(ft_data[group]['artifact']).T * scale
    mat.close()
    return arts

def load_mat(fname, var_name='data'):
    '''
    From kingjr@github
    '''

    # load Matlab/Fieldtrip data
    mat = h5py.File(fname)
    ft_data = mat['data']
    # convert to mne
    n_trial = ft_data['trial'].shape[0]
    n_samps = [mat[ft_data['trial'][t, 0]].shape[0] for t in xrange(n_trial)]
    n_chans = mat[ft_data['trial'][0,0]].shape[1]
    data = np.zeros((sum(n_samps),n_chans))
    time = np.zeros((sum(n_samps),))
    offset = 0
    for trial in range(n_trial):
        # Read MEG data
        trial_data = mat[ft_data['trial'][trial,0]]
        length = trial_data.shape[0]
        data[offset:offset+length, :] = trial_data
        # Read time data
        t = np.array(mat[ft_data['time'][trial,0]]).ravel()
        time[offset:offset+length] = t
        offset+=length

    sfreq = float(ft_data['fsample'][0,0])

    # Encode channels
    chan_names = [''.join(unichr(c) for c in mat[ft_data['label'][0,l]]) for l in range(ft_data['label'].shape[1])]
    chan_names = [ch+'-2622' if ch.startswith('M') else ch for ch in chan_names]
    chan_types = ['mag' if ch.startswith('M') else 'misc' for ch in chan_names]

    info = create_info(chan_names, sfreq, chan_types)
    raw = RawArray(data.T, info, verbose=False)
    event_fields = ['start', 'correct', 'correct_v', 'noise_sigma', 'noise_sigma_v',
        'ref_onset', 'ref_offset', 'stim_onset', 'cc1', 'cc2', 'cc3',
        'cc4', 'cc5', 'cc6', 'cc7', 'cc8', 'cc9', 'cc10', 'stim_offset', 'response',
        'response_v', 'feedback', 'feedback_v', 'end', 'trial']
    trl = pd.DataFrame(np.array(ft_data['trialinfo']).T, columns=event_fields)
    return raw, data, trl, time
