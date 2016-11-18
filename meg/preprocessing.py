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
from pylab import *
import pandas as pd
import glob
from itertools import product
from conf_analysis.behavior import metadata, empirical
from os.path import basename, join, isfile
from conf_analysis.meg.tools import hilbert
from conf_analysis.meg import artifacts
import logging
from joblib import Memory
import cPickle
import os

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
    Each succesfull session consists out of five blocks, yet a sessions MEG data
    file sometimes contains more. This happens, for example, when a block is
    restarted. 'block_in_raw' refers to the actual block in a raw file, and
    block_in_experiment to the block in the metadata that block_in_raw should be
    mapped to. block_in_experiment will be used for saving.
    '''

    try:

        art_fname = metadata.get_epoch_filename(snum, session,
            block_in_experiment, None, 'artifacts')

        data = empirical.load_data()
        data = empirical.data_cleanup(data)

        filename = metadata.get_raw_filename(snum, session)
        raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
        trials = blocks(raw)
        if  not (block_in_raw in unique(trials['block'])):
            err_msg = 'Error when processing %i, %i, %i, %i, data file = %s'%(snum, session, block_in_raw, block_in_experiment, filename)
            raise RuntimeError(err_msg)

        #block_in_raw, block_in_experiment = block_map

        # Load data and preprocess it.
        logging.info('Loading block of data: %s; block: %i'%(filename, block_in_experiment))
        r, r_id = load_block(raw, trials, block_in_raw)
        logging.info('Starting artifact detection')
        r, ants, artdefs = preprocess_block(r)
        logging.info('Aligning meta data')
        meta, timing = get_meta(data, r, snum, block_in_experiment)

        artdefs['id'] = r_id
        cPickle.dump(artdefs, open(art_fname, 'w'), protocol=2)

        for epoch, event, (tmin, tmax) in zip(
                                             ['stimulus', 'response', 'feedback'],
                                             ['stim_onset_t', 'button_t', 'meg_feedback_t'],
                                             [(-.2, 1.5), (-1.5, .5), (-.5, .5)]
                                             ):

            logging.info('Processing epoch: %s' %epoch)
            m, s = get_epoch(r, meta, timing,
                                           event=event, epoch_time=(tmin, tmax),
                                           base_event='stim_onset_t', base_time=(-.2, 0))

            if len(s)>0:
                epo_fname = metadata.get_epoch_filename(snum, session,
                                            block_in_experiment, epoch, 'fif')
                epo_metaname = metadata.get_epoch_filename(snum, session,
                                            block_in_experiment, epoch, 'meta')
                s = s.resample(600, npad='auto')
                s.save(epo_fname)
                m.to_hdf(epo_metaname, 'meta')
    except MemoryError:
        print snum, session, block_in_raw, block_in_experiment
        raise RuntimeError('MemoryError caught in one block ' + str(snum) + ' ' + str(session) + ' ' + str(block_in_raw) + ' ' + str(block_in_experiment) )
    return 'Finished', snum, session, block_in_experiment


def get_block_meta(snum, session, block_in_raw, block_in_experiment):
    data = empirical.load_data()
    data = empirical.data_cleanup(data)
    filename = metadata.get_raw_filename(snum, session)
    raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
    trials = blocks(raw)
    if  not (block_in_raw in unique(trials['block'])):
        err_msg = 'Error when processing %i, %i, %i, %i, data file = %s'%(snum, session, block_in_raw, block_in_experiment, filename)
        raise RuntimeError(err_msg)
    r, r_id = load_block(raw, trials, block_in_raw)
    meta, timing = get_meta(data, r, snum, block_in_experiment)
    return pd.concat((meta, timing), axis=1)


def blocks(raw, full_file_cache=False):
    '''
    Return a dictionary that encodes information about trials in raw.
    '''
    if full_file_cache:
        trigs, buts = get_events_from_file(raw.info['filename'])
    else:
        trigs, buts = get_events(raw)
    es, ee, trl, bl = metadata.define_blocks(trigs)
    return {'start':es, 'end':ee, 'trial':trl, 'block':bl}


def load_block(raw, trials, block):
    '''
    Crop a block of trials from raw file.
    '''
    start = trials['start'][trials['block']==block].min()
    end = trials['end'][trials['block']==block].max()
    r = raw.copy().crop(max(0, raw.times[start]-5), min(raw.times[-1], raw.times[end]+5))
    r_id = {'filename':r.info['filename'], 'first_samp':r.first_samp}
    r.load_data()
    return r, r_id


def preprocess_block(raw):
    '''
    Apply artifact detection to a block of data.
    '''
    ab = artifacts.annotate_blinks(raw)
    am, zm = artifacts.annotate_muscle(raw)
    ac, zc, d = artifacts.annotate_cars(raw)
    ar, zj = artifacts.annotate_jumps(raw)
    ants = artifacts.combine_annotations([x for x in  [ab, am, ac, ar] if x is not None])
    ants.onset += raw.first_samp/raw.info['sfreq']
    raw.annotations = ants
    artdef = {'muscle':zm, 'cars':(zc, d), 'jumps':zj}
    return raw, ants, artdef


def get_meta(data, raw, snum, block):
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
    trigs, buts = get_events(raw)
    es, ee, trl, bl = metadata.define_blocks(trigs)

    megmeta = metadata.get_meta(trigs, es, ee, trl, bl,
                                metadata.fname2session(raw.info['filename']), snum)
    assert len(unique(megmeta.snum)==1)
    assert len(unique(megmeta.day)==1)
    assert len(unique(megmeta.block_num)==1)

    dq = data.query('snum==%i & day==%i & block_num==%i'%(megmeta.snum.ix[0], megmeta.day.ix[0], block))
    #dq.loc[:, 'trial'] = data.loc[:, 'trial']
    trial_idx = np.in1d(dq.trial, unique(megmeta.trial))
    dq = dq.iloc[trial_idx, :]
    dq = dq.set_index(['day', 'block_num', 'trial'])
    megmeta = metadata.correct_recording_errors(megmeta)
    megmeta.loc[:, 'block_num'] = block

    megmeta = megmeta.set_index(['day', 'block_num', 'trial'])
    del megmeta['snum']
    meta = pd.concat([megmeta, dq], axis=1)
    meta = metadata.cleanup(meta) # Performs some alignment checks
    cols = [x for x in meta.columns if x[-2:] == '_t']
    timing = meta.loc[:, cols]
    return meta.drop(cols, axis=1), timing


def get_epoch(raw, meta, timing,
              event='stim_onset_t', epoch_time=(-.2, 1.5),
              base_event='stim_onset_t', base_time=(-.2, 0),
              epoch_label='hash'):
    '''
    Cut out epochs from raw data and apply baseline correction.

    Parameters
    ----------
    raw : raw data
    meta, timing : Dataframes that contain meta and timing information
    event : Column in timing that contains event onsets in sample time
    epoch_time : (start, end) in sec. relative to event onsets defined by 'event'
    base_event : Column in timing that contains baseline onsets in sample time
    base_time : (start, end) in sec. relative to baseline onset
    epoch_label : Column in meta that contains epoch labels.
    '''
    joined_meta = pd.concat([meta, timing], axis=1)
    ev = metadata.mne_events(joined_meta, event, epoch_label)
    eb = metadata.mne_events(joined_meta, base_event, epoch_label)
    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, exclude='bads')

    base = mne.Epochs(raw, eb, tmin=base_time[0], tmax=base_time[1], baseline=None, picks=picks,
            reject_by_annotation=True)
    stim_period = mne.Epochs(raw, ev, tmin=epoch_time[0], tmax=epoch_time[1], baseline=None, picks=picks,
            reject_by_annotation=True)
    base.load_data()
    stim_period.load_data()
    stim_period, dl = apply_baseline(stim_period, base)
    # Now filter raw object to only those left.
    sei = stim_period.events[:, 2]
    meta = meta.reset_index().set_index('hash').loc[sei]
    return meta, stim_period


def concat(raws, metas, timings):
    '''
    Concatenate a set of raw objects and apply offset to meta to
    keep everything in sync. Should allow to load all sessions of
    a subject. Can then crop to parallelize.
    '''
    raws = [r.copy() for r in raws]
    offsets = np.cumsum([0]+[len(raw) for raw in raws])
    raw = raws[::-1].pop()
    raw.append(raws, preload=False)
    timings = [timing+offset for timing, offset in zip(timings, offsets)]
    for t in timings:
        print t.stim_onset_t.min()
    timings = pd.concat(timings)
    metas = pd.concat(metas)
    return raw, metas, timings


def apply_baseline(epochs, baseline):
    drop_list = []
    for epoch, orig in enumerate(epochs.selection):
        # Find baseline epoch for this.
        base = where(baseline.selection==orig)[0]
        if len(base) == 0:
            # Reject this one.
            drop_list.append(epoch)
        else:
            base_val = squeeze(baseline._data[base, :, :]).mean(1)
            epochs._data[epoch, :, :] -= base_val[:, newaxis]

    return epochs.drop(drop_list), drop_list


@memory.cache
def get_events_from_file(filename):
    raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
    buttons = mne.find_events(raw, 'UPPT002', shortest_event=1)
    triggers = mne.find_events(raw, 'UPPT001', shortest_event=1)
    return triggers, buttons

def get_events(raw):
    buttons = mne.find_events(raw, 'UPPT002', shortest_event=1)
    triggers = mne.find_events(raw, 'UPPT001', shortest_event=1)
    return triggers, buttons


def load_epochs(filenames):
    return [mne.read_epochs(f) for f in filenames]


def load_meta(filenames):
    return [pd.read_hdf(f, 'meta') for f in filenames]


def concatenate_epochs(epochs, metas):
    '''
    Concatenate a list of epoch and meta objects and set their dev_head_t projection to
    that of the first epoch.
    '''
    dev_head_t = epochs[0].info['dev_head_t']
    index_cnt = 0
    epoch_arrays = []
    processed_metas = []
    for e, m in zip(epochs, metas):
        e.info['dev_head_t'] = dev_head_t
        processed_metas.append(m)
        e = mne.epochs.EpochsArray(e._data, e.info, events=e.events, tmin=e.tmin)
        epoch_arrays.append(e)
    return mne.concatenate_epochs(epoch_arrays), pd.concat(processed_metas)


@memory.cache
def get_epochs_for_subject(snum, epoch):
    from itertools import product

    metas = [metadata.get_epoch_filename(snum, session, block, epoch, 'meta')
                for session, block in product(range(4), range(5))
                    if os.path.isfile(metadata.get_epoch_filename(snum, session, block, epoch, 'meta'))]
    data = [metadata.get_epoch_filename(snum, session, block, epoch, 'fif')
                for session, block in product(range(4), range(5))
                    if os.path.isfile(metadata.get_epoch_filename(snum, session, block, epoch, 'fif'))]
    assert len(metas)==len(data)
    meta = load_meta(metas)
    data = load_epochs(data)
    return concatenate_epochs(data, meta)


def combine_annotations(annotations, first_samples, last_samples, sfreq):
    '''
    Concatenate a list of annotations objects such that annotations
    stay in sync with the output of mne.concatenate_raws.

    This function assumes that annotations objects come from different raw objects
    that are to be concatenated. In this case the concatenated raw object retains
    the first sample of the first raw object and then treats the data as
    continuous. In contrast, the annotation onsets already shifted by each individual
    raw object's first sample to be in sync. When concatenting annotations this
    needs to be taken into account.

    Parameters
    ----------
    annotations : list of annotations objects, shape (n_objects,)
    first_samples : list of ints, shape (n_objects,)
        First sample of each annotations' raw object.
    last_samples : list of ints, shape (n_objects,)
        Last sample of each annotations' raw object.
    sfreq : int
        Sampling frequency of data in raw objects.
    '''
    if all([ann is None for ann in annotations]):
        return None
    durations = [(1+l-f)/sfreq for f, l in zip(first_samples, last_samples)]
    offsets = np.cumsum([0] + durations[:-1])

    onsets = [(ann.onset-(fs/sfreq))+offset
                        for ann, fs, offset in zip(annotations, first_samples, offsets) if ann is not None]

    if len(onsets) == 0:
        return mne.annotations.Annotations(onset=[], duration=[], description=None)

    onsets = np.concatenate(onsets) + (first_samples[0]/sfreq)
    return mne.annotations.Annotations(onset=onsets,
        duration=np.concatenate([ann.duration for ann in annotations]),
        description=np.concatenate([ann.description for ann in annotations]))
