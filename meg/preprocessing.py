'''
Preprocess an MEG data set.

The idea for preprocessing MEG data is modelled around a few aspects of the
confidence data set:
    1. Each MEG dataset is accompanied by a DataFrame that contains metadata for
       each trial.
    2. Trial meta data can be matched to the MEG data by appropriate triggers.
    3. Each MEG datafile contains several blocks of data that can be processed
       independently.

This leads to the following design:
    1. MEG is cut into recording blocks and artifact detection is carried out
    2. Each processed block is matched to meta data and timing (event on- and offsets)
       data is extracted from the MEG and aligned with the behavioral data.
    3. Data is epoched. Sync with meta data is guaranteed by a unique key for
       each trial that is stored along with the epoched data.
'''

import mne
from numpy import *
import numpy as np
import pandas as pd
import glob
from itertools import product
from conf_analysis.behavior import metadata, empirical
from os.path import basename, join, isfile
from conf_analysis.meg.tools import hilbert
import logging
from joblib import Memory
import pickle
import os

from pymeg import preprocessing as pymegprepr

memory = Memory(cachedir=metadata.cachedir)


def one_block(snum, session, block_in_raw, block_in_experiment):
    '''
    Preprocess a single block and save results.

    Parameters
    ----------
        snum, session : int
    Subject number and session number
        raw : mne.io.Raw object
    Raw data for an entire session of a subject.
        block_in_raw, block_in_experiment : int
    Each succesfull session consists out of five blocks, yet a sessions MEG
    data file sometimes contains more. This happens, for example, when a block
    is restarted. 'block_in_raw' refers to the actual block in a raw file, and
    block_in_experiment to the block in the metadata that block_in_raw should
    be mapped to. block_in_experiment will be used for saving.
    '''

    try:

        art_fname = metadata.get_epoch_filename(
            snum, session,
            block_in_experiment,
            None, 'artifacts')

        data = empirical.load_data()
        data = empirical.data_cleanup(data)

        filename = metadata.get_raw_filename(snum, session)
        raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
        trials = blocks(raw)
        if not (block_in_raw in unique(trials['block'])):
            err_msg = 'Error when processing %i, %i, %i, %i, data file = %s' % (
                snum, session, block_in_raw, block_in_experiment, filename)
            raise RuntimeError(err_msg)

        # Load data and preprocess it.
        logging.info('Loading block of data: %s; block: %i' %
                     (filename, block_in_experiment))
        r, r_id = load_block(raw, trials, block_in_raw)
        r_id['filnemae'] = filename
        print('Working on:', filename, block_in_experiment, block_in_raw)
        logging.info('Starting artifact detection')

        r, ants, artdefs = pymegprepr.preprocess_block(r)
        print('Notch filtering')
        r.notch_filter(np.arange(50, 251, 50))
        logging.info('Aligning meta data')
        meta, timing = get_meta(data, r, snum, block_in_experiment, filename)
        idx = np.isnan(meta.response.values)
        meta = meta.loc[~idx, :]
        timing = timing.loc[~idx, :]
        artdefs['id'] = r_id
        filenames = []
        for epoch, event, (tmin, tmax), (rmin, rmax) in zip(
                ['stimulus', 'response', 'feedback'],
                ['stim_onset_t', 'button_t',
                 'meg_feedback_t'],
                [(-.75, 1.5), (-1.5, 1), (-1, 1)],
                [(0, 1), (-1, 0.5), (-0.5, 0.5)]):

            logging.info('Processing epoch: %s' % epoch)
            m, s = pymegprepr.get_epoch(r, meta, timing,
                                        event=event, epoch_time=(tmin, tmax),
                                        reject_time=(rmin, rmax),
                                        base_event='stim_onset_t', base_time=(-.2, 0))

            if len(s) > 0:
                epo_fname = metadata.get_epoch_filename(snum, session,
                                                        block_in_experiment, epoch, 'fif')
                epo_metaname = metadata.get_epoch_filename(snum, session,
                                                           block_in_experiment, epoch, 'meta')
                s = s.resample(600, npad='auto')
                s.save(epo_fname)
                m.to_hdf(epo_metaname, 'meta')
                r_id[epoch] = len(s)
                filenames.append(epo_fname)
        pickle.dump(artdefs, open(art_fname, 'w'), protocol=2)

    except MemoryError:
        print(snum, session, block_in_raw, block_in_experiment)
        raise RuntimeError('MemoryError caught in one block ' + str(snum) + ' ' + str(
            session) + ' ' + str(block_in_raw) + ' ' + str(block_in_experiment))
    return 'Finished', snum, session, block_in_experiment, filenames


def get_block_meta(snum, session, block_in_raw, block_in_experiment):
    data = empirical.load_data()
    data = empirical.data_cleanup(data)
    filename = metadata.get_raw_filename(snum, session)
    raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
    trials = blocks(raw)
    if not (block_in_raw in unique(trials['block'])):
        err_msg = 'Error when processing %i, %i, %i, %i, data file = %s' % (
            snum, session, block_in_raw, block_in_experiment, filename)
        raise RuntimeError(err_msg)
    r, _ = load_block(raw, trials, block_in_raw)
    meta, timing = get_meta(data, r, snum, block_in_experiment, filename)
    return pd.concat((meta, timing), axis=1)


def blocks(raw, full_file_cache=False):
    '''
    Return a dictionary that encodes information about trials in raw.
    '''
    if full_file_cache:
        trigs, buts = pymegprepr.get_events_from_file(raw.info['filename'])
    else:
        trigs, buts = pymegprepr.get_events(raw)
    es, ee, trl, bl = metadata.define_blocks(trigs)
    return {'start': es, 'end': ee, 'trial': trl, 'block': bl}


def load_block(raw, trials, block):
    '''
    Crop a block of trials from raw file.
    '''
    start = trials['start'][trials['block'] == block].min()
    end = trials['end'][trials['block'] == block].max()
    r = raw.copy().crop(
        max(0, raw.times[start] - 5), min(raw.times[-1], raw.times[end] + 5))
    r_id = {'first_samp': r.first_samp}
    r.load_data()
    return r, r_id


def get_meta(data, raw, snum, block, filename):
    '''
    Return meta and timing data for a raw file and align it with behavioral data.

    Parameters
    ----------
    data : DataFrame
        Contains trial meta data from behavioral responses.
    raw : mne.io.raw objects
        MEG data that needs to be aligned to the behavioral data.
    snum : int
        Subject number that this raw object corresponds to.
    block : int
        Block within recording that this raw object corresponds to.

    Note: Data is matched agains the behavioral data with snum, recording, trial
    number and block number. Since the block number is not encoded in MEG data it
    needs to be passed explicitly. The order of responses is encoded in behavioral
    data and MEG data and is compared to check for alignment.
    '''
    trigs, buts = pymegprepr.get_events(raw)
    es, ee, trl, bl = metadata.define_blocks(trigs)

    megmeta = metadata.get_meta(trigs, es, ee, trl, bl,
                                metadata.fname2session(filename), snum)
    assert len(unique(megmeta.snum) == 1)
    assert len(unique(megmeta.day) == 1)
    assert len(unique(megmeta.block_num) == 1)

    dq = data.query('snum==%i & day==%i & block_num==%i' %
                    (megmeta.snum.ix[0], megmeta.day.ix[0], block))
    #dq.loc[:, 'trial'] = data.loc[:, 'trial']
    trial_idx = np.in1d(dq.trial, unique(megmeta.trial))
    dq = dq.iloc[trial_idx, :]
    dq = dq.set_index(['day', 'block_num', 'trial'])
    megmeta = metadata.correct_recording_errors(megmeta)
    megmeta.loc[:, 'block_num'] = block

    megmeta = megmeta.set_index(['day', 'block_num', 'trial'])
    del megmeta['snum']
    meta = pd.concat([megmeta, dq], axis=1)
    meta = metadata.cleanup(meta)  # Performs some alignment checks
    cols = [x for x in meta.columns if x[-2:] == '_t']
    timing = meta.loc[:, cols]
    return meta.drop(cols, axis=1), timing


@memory.cache
def get_epochs_for_subject(snum, epoch):
    from itertools import product

    metas = [metadata.get_epoch_filename(snum, session, block, epoch, 'meta')
             for session, block in product(list(range(4)), list(range(5)))
             if os.path.isfile(metadata.get_epoch_filename(snum, session, block, epoch, 'meta'))]
    data = [metadata.get_epoch_filename(snum, session, block, epoch, 'fif')
            for session, block in product(list(range(4)), list(range(5)))
            if os.path.isfile(metadata.get_epoch_filename(snum, session, block, epoch, 'fif'))]
    assert len(metas) == len(data)
    meta = pymegprepr.load_meta(metas)
    data = pymegprepr.load_epochs(data)
    channels = [set(e.ch_names) for e in data]
    channels = list(channels[0].intersection(*channels[1:]))
    data = [e.drop_channels([x for x in e.ch_names if x not in channels])
            for e in data]
    return pymegprepr.concatenate_epochs(data, meta)


def get_meta_for_subject(snum, epoch):
    metas = [metadata.get_epoch_filename(snum, session, block, epoch, 'meta')
             for session, block in product(list(range(4)), list(range(5)))
             if os.path.isfile(
                 metadata.get_epoch_filename(
                     snum, session, block, epoch, 'meta'))]
    meta = pymegprepr.load_meta(metas)
    return pd.concat(meta)
