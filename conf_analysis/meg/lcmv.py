import logging
import mne
import numpy as np
import os

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing
from conf_analysis.meg import source_recon as sr

from joblib import Memory

import pandas as pd

from os import makedirs
from os.path import join
from pymeg import lcmv as pymeglcmv
from pymeg import source_reconstruction as pymegsr


memory = Memory(cachedir=metadata.cachedir)
path = '/home/nwilming/conf_meg/sr_labeled/'


def set_n_threads(n):
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    os.environ['OMP_NUM_THREADS'] = str(n)


def submit():
    from pymeg import parallel
    from itertools import product
    for subject, session, epoch, signal in product(
            [3,6,7,8,15], range(4), ['stimulus', 'response'],
            ['LF']):

        parallel.pmap(
            extract, [(subject, session, epoch, signal)],
            walltime='10:00:00', memory=40, nodes=1, tasks=4,
            name='SR' + str(subject) + '_' + str(session) + epoch,
            ssh_to=None)


def lcmvfilename(subject, session, signal, epoch_type, chunk=None):
    try:
        makedirs(path)
    except:
        pass
    if chunk is None:
        filename = 'S%i-SESS%i-%s-%s-lcmv.hdf' % (
            subject, session, epoch_type, signal)
    else:
        filename = 'S%i-SESS%i-%s-%s-chunk%i-lcmv.hdf' % (
            subject, session, epoch_type, signal, chunk)
    return join(path, filename)


def get_stim_epoch(subject, session):
    epochs, meta = preprocessing.get_epochs_for_subject(subject,
                                                        'stimulus',
                                                        sessions=session)
    epochs = epochs.pick_channels(
        [x for x in epochs.ch_names if x.startswith('M')])
    id_time = (-0.25 <= epochs.times) & (epochs.times <= 0)
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]
    data_cov = pymeglcmv.get_cov(epochs, tmin=0, tmax=1.35)
    return data_cov, epochs


def get_response_epoch(subject, session):
    epochs, meta = preprocessing.get_epochs_for_subject(subject,
                                                        'stimulus',
                                                        sessions=session)
    epochs = epochs.pick_channels(
        [x for x in epochs.ch_names if x.startswith('M')])
    response, meta = preprocessing.get_epochs_for_subject(subject,
                                                          'response',
                                                          sessions=session)
    response = response.pick_channels(
        [x for x in response.ch_names if x.startswith('M')])
    # Find trials that are present in both time periods
    overlap = list(
        set(epochs.events[:, 2]).intersection(
            set(response.events[:, 2])))
    epochs = epochs[[str(l) for l in overlap]]
    response = response[[str(l) for l in overlap]]
    id_time = (-0.25 <= epochs.times) & (epochs.times <= 0)
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]

    response._data = (
        response._data - means[:, :, np.newaxis])

    # Now also baseline stimulus period
    epochs, meta = preprocessing.get_epochs_for_subject(subject,
                                                        'stimulus',
                                                        sessions=session)
    epochs = epochs.pick_channels(
        [x for x in epochs.ch_names if x.startswith('M')])
    id_time = (-0.25 <= epochs.times) & (epochs.times <= 0)
    means = epochs._data[:, :, id_time].mean(-1)
    epochs._data -= means[:, :, np.newaxis]
    data_cov = pymeglcmv.get_cov(epochs, tmin=0, tmax=1.35)
    return data_cov, response


def extract(subject, session, epoch_type='stimulus', signal_type='BB',
            BEM='three_layer', debug=False, chunks=100, njobs=4):
    mne.set_log_level('WARNING')
    pymeglcmv.logging.getLogger().setLevel(logging.INFO)
    set_n_threads(1)

    logging.info('Reading stimulus data')
    if epoch_type == 'stimulus':
        data_cov, epochs = get_stim_epoch(subject, session)
    elif epoch_type == 'response':
        data_cov, epochs = get_response_epoch(subject, session)
    else:
        raise RuntimeError('Did not recognize epoch')

    logging.info('Setting up source space and forward model')

    forward, bem, source = sr.get_leadfield(subject, session, BEM)

    labels = pymegsr.get_labels('S%02i' % subject)

    # Now chunk Reconstruction into blocks of ~100 trials to save Memory
    fois = np.arange(10, 150, 5)
    lfois = np.arange(1, 10, 1)
    tfr_params = {
        'F': {'foi': fois, 'cycles': fois * 0.1, 'time_bandwidth': 2,
              'n_jobs': 1, 'est_val': fois, 'est_key': 'F'},
        'LF': {'foi': lfois, 'cycles': lfois * 0.25, 'time_bandwidth': 2,
               'n_jobs': 1, 'est_val': lfois, 'est_key': 'LF'}
    }

    events = epochs.events[:, 2]
    data = []
    filters = pymeglcmv.setup_filters(epochs.info, forward, data_cov,
                                      None, labels)
    set_n_threads(1)

    for i in range(0, len(events), chunks):
        filename = lcmvfilename(
            subject, session, signal_type, epoch_type, chunk=i)
        if os.path.isfile(filename):
            continue
        if signal_type == 'BB':
            logging.info('Starting reconstruction of BB signal')
            M = pymeglcmv.reconstruct_broadband(
                filters, epochs.info, epochs._data[i:i + chunks],
                events[i:i + chunks],
                epochs.times, njobs=1)
        else:
            logging.info('Starting reconstruction of TFR signal')
            M = pymeglcmv.reconstruct_tfr(
                filters, epochs.info, epochs._data[i:i + chunks],
                events[i:i + chunks], epochs.times,
                est_args=tfr_params[signal_type],
                njobs=4)
        M.to_hdf(filename, 'epochs')
    set_n_threads(njobs)
    
