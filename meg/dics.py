
import numpy as np
import mne
from conf_analysis.meg import source_recon as sr
from conf_analysis.behavior import metadata
from pymeg import preprocessing as pymegprepr, tfr

from copy import deepcopy

from scipy import linalg

from mne.beamformer._lcmv import (_prepare_beamformer_input, _setup_picks,
                                  _reg_pinv)
from mne.time_frequency import csd_epochs as mne_csd_epochs
from mne.externals import six
from joblib import Memory
from joblib import Parallel, delayed

import time as clock
from glob import glob
import pandas as pd

memory = Memory(cachedir=metadata.cachedir)


@memory.cache
def get_tfr(subject, n_blocks=None):
    from mne.time_frequency.tfr import read_tfrs
    from mne.time_frequency import EpochsTFR
    files = glob('/home/nwilming/conf_meg/S%i/SUB%i*stimulus*-tfr.h5' %
                 (subject, subject))
    if n_blocks is not None:
        files = files[:n_blocks]
    etfr = [read_tfrs(files.pop())[0]]
    for fname in files:
        etfr.append(read_tfrs(fname)[0])
    data = np.concatenate([e.data for e in etfr], 0)
    return EpochsTFR(etfr[0].info, data, etfr[0].times, etfr[0].freqs)


def get_metas_for_tfr(subject):
    files = glob('/home/nwilming/conf_meg/S%i/SUB%i*stimulustfr.hdf5' %
                 (subject, subject))
    metas = []
    for f in files:
        sub, sess, block, _ = f.split('/')[-1].split('_')
        sess = int(sess[1:])
        block = int(block[1:])
        metas.append(get_meta_for_block(subject, sess, block))
    return pd.concat(metas)


def get_meta_for_block(subject, session, block):
    meta = metadata.get_epoch_filename(
        subject, session, block, 'stimulus', 'meta')
    meta = pymegprepr.load_meta([meta])
    return meta[0]


def get_tfr_filename_for_trial(meta, trial):
    from itertools import product
    meta = meta.loc[trial]
    blocks = meta.block_num.values
    session = meta.session_num.values
    subject = meta.snum.values

    combinations = set(zip(subject, session, blocks))
    return [('/home/nwilming/conf_meg/S%i/SUB%i_S%i_B%i_stimulustfr.hdf5' %
             (sub, sub, sess - 1, block),
             meta.query('snum==%i & session_num==%i & block_num==%i' %
                        (sub, sess, block)))
            for sub, sess, block in combinations]


def get_tfr_array(meta, freq=(0, 100), channel=None, tmin=None, tmax=None,
                  baseline=None):
    '''
    Load many saved tfrs and return as a numpy array.

    Inputs
    ------
        filenames: List of TFR filenames
        freq:  tuple that specifies which frequencies to pull from TFR file.
        channel: List of channels to include
        tmin & tmax: which time points to include
        baseline: If func it will be applied to each TFR file that is being loaded.
    '''
    filenames = get_tfr_filename_for_trial(meta, meta.index.values)
    fname, block = filenames[0]
    out = tfr.read_chunked_hdf(fname, freq=freq, channel=channel,
                               tmin=tmin, tmax=tmax, epochs=block.index.values)
    freqs, times, events = out['freqs'], out['times'], out['events']
    dfs = [out['data']]
    read_epochs = [events]
    for fname, block in filenames[1:]:
        try:
            out = tfr.read_chunked_hdf(fname, freq=freq, channel=channel,
                                       tmin=tmin, tmax=tmax,
                                       epochs=block.index.values)
            f, t, e = out['freqs'], out['times'], out['events']
            assert(all(freqs == f))
            assert(all(times == t))
            read_epochs.append(e)
            dfs.append(out['data'])
        except RuntimeError as e:
            print e
            assert e.msg.contains('None of the requested epochs are')
    return freqs, times, np.concatenate(read_epochs), np.concatenate(dfs, 0)


@memory.cache
def make_csds(epochs, freqs, f, times, f_smooth, t_smooth, subject,
              n_jobs=10):
    '''
    Compute Beamformer filters for time-points in TFR.
    '''
    id_freq = np.argmin(np.abs(freqs - f))
    f = freqs[id_freq]
    fmin, fmax = f - f_smooth, f + f_smooth
    forward, bem, source, trans = sr.get_leadfield(subject)
    idx = ((times[0] + t_smooth) < times) & (times < (times[-1] - t_smooth))

    print "Computing noise csd with t_smooth:", t_smooth
    noise_csd = get_noise_csd(epochs, fmin, fmax, t_smooth)

    data_csds = []
    with Parallel(n_jobs=n_jobs) as parallel:
        filters = parallel(
            delayed(one_csd)(i, epochs, forward, noise_csd, time,
                             fmin, fmax, t_smooth)
            for i, time in enumerate(times[idx]))

        print 'Done with CSDds'
    return f, filters


def one_csd(i, epochs, forward, noise_csd, time, fmin, fmax,
            t_smooth):

    start = clock.time()
    tmin = time - t_smooth
    tmax = time + t_smooth
    epochs.crop(tmin, tmax)
    data_csd = mne_csd_epochs(epochs, 'multitaper',
                              fmin=fmin,
                              fmax=fmax,
                              fsum=True,
                              tmin=time - t_smooth,
                              tmax=time + t_smooth)
    print i, 'CSD - Filter took', np.around(clock.time() - start, 3)
    del epochs._data
    info = epochs.info
    t = (tmax + tmin) / 2.
    print i, 'CSD + Filter took', np.around(clock.time() - start, 3)
    return time, data_csd


@memory.cache
def get_noise_csd(epochs, fmin, fmax, t_smooth):
    return mne_csd_epochs(epochs, 'multitaper', fmin, fmax,
                          fsum=True, tmin=0.75 - 2 * t_smooth,
                          tmax=0.75)


def apply_dics_filter(filters, F, meta, filename, subject, n_jobs=1):
    '''
    Apply beamformer to TFR data

    Filters is a dictionary of CSDs, where keys are timepoints.
    f is the frequency of interest
    meta is a meta structure giving information about trials
    filename indicates which memmap file to use for storing source info

    '''
    num_epochs = meta.shape[0]
    forward, bem, source, trans = sr.get_leadfield(subject)
    n_source = np.sum([s['nuse'] for s in source])
    sp_shape = (n_source, len(filters.keys()), num_epochs)
    # memmap results
    source_pow = np.memmap(filename, dtype='float32', mode='w+',
                           shape=sp_shape)
    del forward, bem, source, trans

    print 'Applying in parallel'

    def get_jobs():
        time = np.sort(filters.keys())
        for tidx, t in enumerate(time):
            csd = filters[t]
            args = (source_pow, meta, (F, F),
                    (t, t), csd, subject, 0, tidx)
            yield delayed(apply_one_filter)(*args)

    epoch_order = Parallel(n_jobs=n_jobs)(get_jobs())

    return source_pow, epoch_order


def apply_one_filter(source_pow, meta, freq, time, csd, subject, offset, tidx):
    forward, bem, source, trans = sr.get_leadfield(subject)
    # Load source data
    freqs, times, epochs, tfrdata = get_tfr_array(meta, freq=freq,
                                                  tmin=time[0],
                                                  tmax=time[1])
    tfrdata = tfrdata.squeeze() # now num_epochs x channels

    A = dics_filter(forward, csd)
    epochs_order = []
    indices = []
    for i, (epoch, Xsensor) in enumerate(zip(epochs, tfrdata), offset):
        Xsource = np.dot(A, Xsensor)  # 8k = (8k x 269) * (269,)
        source_pow[:, tidx, i] = Xsource * np.conj(Xsource)
        epochs_order.append(epoch)
        indices.append(i)
    print 'Done with %f'%time[0]
    return indices, epochs_order


@memory.cache
def single_trial_memmap_shape(subject, shape):
    '''
    Return the shape of the memmap array that is used for saving the single
    trial power estimates.
    Shape is: Number of sources, number of time points, number of epochs
    '''
    tfrepochs = get_tfr(subject, n_blocks=1)
    print(tfrepochs.data.shape)
    n_time = tfrepochs.data.shape[3]
    forward, bem, source, trans = sr.get_leadfield(subject)
    n_source = np.sum([s['nuse'] for s in source])
    n_epochs = shape / (n_time * n_source)
    return n_source, n_epochs, n_time


def stc_from_memmap(data, subject):
    forward, bem, source, trans = sr.get_leadfield(subject)
    verts = [source[0]['vertno'], source[1]['vertno']]
    tmin = -0.75
    tstep = 0.01666667
    stc = mne.SourceEstimate(data, verts, tmin=tmin,
                             tstep=tstep, subject='S%02i' % subject)
    return stc


def extract_label(data, source, label):
    verts = np.concatenate([source[0]['vertno'], source[1]['vertno']])
    assert(len(verts) == data.shape[0])
    idx = np.in1d(verts, label.vertices)
    return data[idx, :, :].mean(0)


def get_label_dataframe(meta, data, source, times, labels):
    '''
    Extract a set of labels from memmapped sources and align with meta.    
    '''
    frames = []
    trials = meta.index.values
    for label in labels:
        d_label = extract_label(data, source, label)
        df = pd.DataFrame(d_label, index=trials, columns=times).stack()
        df.name = label.name
        frames.append(df)
    frames = pd.concat(frames, axis=1)
    frames.columns = [f.replace('lh.wang2015atlas.', '')
                       .replace('rh.wang2015atlas.', '')
                      for f in frames.columns]
    return frames


def power_to_label_dataframe(subject, filename):
    '''
    Convert source power estimates from a subject to a datframe
    '''
    data = np.memmap(filename, dtype='float32', mode='r')
    s = single_trial_memmap_shape(subject, data.shape[0])
    data = np.memmap(filename, dtype='float32', mode='r', shape=s)
    forward, bem, source, trans = sr.get_leadfield(subject)
    meta = get_metas_for_tfr(subject)
    tfrepochs = get_tfr(subject, n_blocks=1)
    times = tfrepochs.times
    labels = sr.get_labels(subject)
    frame = get_label_dataframe(meta, data, source, times, labels)
    outname = filename.replace('memmap', 'labeled.hdf5')
    frame.to_hdf(outname, 'df')


@memory.cache
def dics_filter(forward, data_csd, reg=0.05):
    '''
    forward contains the lead field
    data_csd contains the CSD of interest

    Assume free orientation lead field.
    '''
    Cm = data_csd.data.copy()
    Cm = Cm.real
    #Cm_inv, _ = _reg_pinv(Cm, reg)
    Cm_inv = np.linalg.pinv(Cm + reg * np.eye(Cm.shape[0]))

    source_loc = forward['source_rr']
    source_ori = forward['source_nn']
    As = np.nan * np.ones((source_loc.shape[0], Cm.shape[0]))
    for i, k in enumerate(range(0, source_ori.shape[0], 3)):
        L = forward['sol']['data'][:, k:k + 3]
        A = np.dot(
            np.dot(
                linalg.pinv(
                    np.dot(
                        np.dot(L.T, Cm_inv),
                        L)),
                L.T), Cm_inv)

        # print A.min(), A.max()
        Aw = np.dot(np.dot(A, Cm), np.conj(A).T)
        v, h = np.linalg.eig(Aw)
        # print h, v
        A = np.dot(A.T, h[:, 0])
        As[i, :] = A
    return As
