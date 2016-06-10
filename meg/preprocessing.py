'''
Preprocess an MEG data set.
'''
import mne
from pylab import *
import pandas as pd
import glob
from itertools import product
from conf_analysis.behavior import metadata
from joblib import Memory
from os.path import basename, join, isfile

memory = Memory(cachedir=metadata.cachedir, verbose=0)


def get_epochs(raw, epochs, baseline, tmin=-1, tmax=0, downsample=300, reject=dict(mag=5e-12), picks='meg'):
    if picks is 'meg':
        picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, exclude='bads')

    epochs = mne.Epochs(raw, epochs, tmin=tmin, tmax=tmax,
        baseline=None, picks=picks, reject_by_annotation=True, reject=reject)
    baseline = mne.Epochs(raw, baseline, tmin=tmin, tmax=tmax,
        baseline=None, picks=picks, reject_by_annotation=True, reject=reject)
    epochs.load_data()
    baseline.load_data()

    epochs = apply_baseline(epochs, baseline)
    return epchs.resample(downsample)
    


def get_datasets(data, subject, sessions):
    filenames = [metadata.get_sub_session_rawname('S%02i'%subject, sess) for sess in sessions]
    d  =  [get_dataset(data, filename, subject) for filename in filenames]
    raws = [d[k][0].copy() for k in range(len(d))]
    metas = [d[k][1] for k in range(len(d))]
    timings = [d[k][2] for k in range(len(d))]
    return raws, metas, timings

def from_cache(cachdir):
    if isfile(cachedir):
        # Load from disk
        raw = mne.io.Raw(cachedir)
        meta = pd.read_hdf(cachedir + 'meta.hdf', 'meta')
        timing = pd.read_hdf(cachedir + 'timing.hdf', 'timing')
        return raw, meta, timing
    else:
        raise RuntimeError('Not present in cache: %s'%cachedir)

def to_cache(raw, meta, timing):
    fname = basename(raw.info['filename'])
    cachedir = join(metadata.cachedir, fname)
    raw.save(cachedir + '.raw.fif.gz', overwrite=True)
    meta.to_hdf(cachedir + 'meta.hdf', 'meta')
    timing.to_hdf(cachedir + 'timing.hdf', 'meta')


def get_dataset(data, filename, snum, notch=True):
    '''
    Preprocess a data set and return raw and meta struct. Caches results in an
    intermediate directory (metadata.cachedir).
    '''

    cachedir = join(metadata.cachedir, basename(filename)+'.raw.fif.gz')
    if isfile(cachedir):
        return from_cache(cachedir)

    raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
    raw = annotate_blinks(raw)
    trigs, buts = get_events(raw)
    es, ee, trl, bl = metadata.define_blocks(trigs)
    megmeta = metadata.get_meta(trigs, es, ee, trl, bl,
                                metadata.fname2session(filename), snum)

    assert len(unique(megmeta.snum)==1)
    assert len(unique(megmeta.day)==1)
    data = data.query('snum==%i & day==%i'%(megmeta.snum.ix[0], megmeta.day.ix[0]))
    data = data.set_index(['day', 'block_num', 'trial'])
    megmeta = metadata.correct_recording_errors(megmeta)
    megmeta = megmeta.set_index(['day', 'block_num', 'trial'])
    meta = pd.concat([megmeta, data], axis=1)
    meta = metadata.cleanup(meta)
    cols = [x for x in meta.columns if x[-2:] == '_t']
    timing = meta.loc[:, cols]
    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, exclude='bads')
    dropchans = [x for x in array(raw.ch_names)[picks] if not x.startswith('M')]
    dropchans = dict((k, 'syst') for k in dropchans)
    raw.set_channel_types(dropchans)
    if notch:
        raw.load_data()
        raw.notch_filter(np.arange(50, 251, 50))
    return raw, meta.drop(cols, axis=1), timing


def get_stimulus_epoch(raw, meta, timing, tmin=-.2, tmax=1.5):
    ev = metadata.mne_events(pd.concat([meta, timing], axis=1).loc[~isnan(meta.response), :],
            'stim_onset_t', 'response')
    eb = metadata.mne_events(pd.concat([meta, timing], axis=1).loc[~isnan(meta.response), :],
            'stim_onset_t', 'response')
    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, exclude='bads')
    base = mne.Epochs(raw, eb, tmin=-.2, tmax=.0, baseline=None, picks=picks,
            reject_by_annotation=True, reject=dict(mag=5e-12))
    stimperiod = mne.Epochs(raw, ev, tmin=-.2, tmax=1.2, baseline=None, picks=picks,
            reject_by_annotation=True, reject=dict(mag=5e-12))
    base.load_data()
    stimperiod.load_data()
    stim_period, dl = apply_baseline(stimperiod, base)
    return stim_period


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


def get_events(raw):
    buttons = mne.find_events(raw, 'UPPT002')
    triggers = mne.find_events(raw, 'UPPT001')
    return triggers, buttons


def session(subject, session):
    filename = metadata.get_sub_session_rawname(subject, session)
    raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
    events = mne.find_events(raw, 'UPPT001')
    ts, te, trial, block = metadata.define_blocks(events)
    blocks = unique(block)
    return raw, ts, te, trial, block, blocks, events


def ppd(vieweing_distance=62.5, screen_width=38.0, x_resolution=1450):
    o = tan(0.5*pi/180) *vieweing_distance;
    return 2 * o*x_resolution/screen_width;


def annotate_blinks(raw):
    '''
    Detect blinks and annotate as bad blinks
    '''
    x, y, p = eye_voltage2gaze(raw)
    xpos, ypos = x.ravel()/ppd(), y.ravel()/ppd()
    sc = saccade_detection(xpos, ypos, threshold=10, acc_thresh=2000, Hz=1000)
    blinks, scfilt = blink_detection(xpos, ypos, sc)
    blink_onsets = raw.times[blinks[:,0]]
    blink_durations = raw.times[blinks[:,1]]-raw.times[blinks[:,0]]
    raw.annotations = mne.Annotations(blink_onsets, blink_durations, 'bad blinks')
    return raw

def eye_voltage2gaze(raw, ranges=(-5, 5), screen_x=(0, 1920),
                     screen_y=(0, 1080),
                     ch_mapping={'x':'UADC002-3705', 'y':'UADC003-3705', 'p':'UADC004-3705'}):
    '''
    Convert analog output of EyeLink 1000+ to gaze coordinates.
    '''
    minvoltage, maxvoltage = ranges
    maxrange, minrange = 1., 0.
    screenright, screenleft = screen_x
    screenbottom, screentop = screen_y

    idx = where(array(raw.ch_names) == ch_mapping['x'])[0][0]
    R = (raw[idx, :][0]-minvoltage)/(maxvoltage-minvoltage)
    S = R*(maxrange-minrange)+minrange
    x = S*(screenright-screenleft+1)+screenleft

    idy = where(array(raw.ch_names) == ch_mapping['y'])[0][0]
    R = (raw[idy, :][0]-minvoltage)/(maxvoltage-minvoltage)
    S = R*(maxrange-minrange)+minrange
    y = S*(screenbottom-screentop+1)+screentop

    idp = where(array(raw.ch_names) == ch_mapping['p'])[0][0]
    p = raw[idp, :][0]
    return x, y, p


velocity_window_size = 3
def get_velocity(x, y, Hz):
    '''
    Compute velocity of eye-movements.

    'x' and 'y' specify the x,y coordinates of gaze location. The function
    assumes that the values in x,y are sampled continously at a rate specified
    by 'Hz'.
    '''
    Hz = float(Hz)
    distance = ((np.diff(x) ** 2) +
                (np.diff(y) ** 2)) ** .5
    distance = np.hstack(([distance[0]], distance))
    win = np.ones((velocity_window_size)) / float(velocity_window_size)
    velocity = np.convolve(distance, win, mode='same')
    velocity = velocity / (velocity_window_size / Hz)
    acceleration = np.diff(velocity) / (1. / Hz)
    acceleration = abs(np.hstack(([acceleration[0]], acceleration)))
    return velocity, acceleration


def saccade_detection(x, y, Hz=1000, threshold=30,
                      acc_thresh=2000):
    '''
    Detect saccades in a stream of gaze location samples.

    Coordinates of x,y are assumed to be in degrees.

    Saccades are detect by a velocity/acceleration threshold approach.
    A saccade starts when a) the velocity is above threshold, b) the
    acceleration is above acc_thresh at least once during the interval
    defined by the velocity threshold.
    '''

    velocity, acceleration = get_velocity(x, y, float(Hz))
    saccades = (velocity > threshold)

    borders = np.where(np.diff(saccades.astype(int)))[0] + 1
    if velocity[1] > threshold:
        borders = np.hstack(([0], borders))

    saccade = 0 * np.ones(x.shape)

    saccade_times = []
    # Only count saccades when acceleration also surpasses threshold
    for i, (start, end) in enumerate(zip(borders[0::2], borders[1::2])):
        if sum(acceleration[start:end] > acc_thresh) >= 1:
            saccade[start:end] = 1
            saccade_times.append((start, end))

    return array(saccade_times)


def microssacade_detection(x, y, VFAC):
    if len(x)<5:
        return None
    dt = 1/1000.
    kernel = array([1., 1., 0., -1., -1.])
    vx = convolve(x, kernel, mode='same')/(6*dt)
    vy = convolve(y, kernel, mode='same')/(6*dt)
    msdx = sqrt( median((vx-median(vx))**2))
    msdy = sqrt( median((vy-median(vy))**2))
    radiusx = VFAC*msdx
    radiusy = VFAC*msdy
    test = (vx/radiusx)**2 + (vy/radiusy)**2
    borders = np.where(np.diff((test>1).astype(int)))[0] + 1
    if test[0] > 1:
        borders = np.hstack(([0], borders))
    if test[-1] > 1:
        borders = np.hstack((borders, [len(x)]))

    borders = borders.reshape(len(borders)/2, 2)
    return borders


def blink_detection(x, y, saccades):
    '''
    A blink is everything that is surrounded by two saccades and  period in
    between where the eye is off screen.
    '''
    rm_sac = (saccades[:,0]*0).astype(bool)
    blinks = []
    skipnext=False
    for i, ((pss, pse), (nss, nse)) in enumerate(zip(saccades[:-1], saccades[1:])):
        if skipnext:
            skipnext=False
            continue
        xavg = x[pse:nss].mean()
        yavg = y[pse:nss].mean()

        if (xavg>40) and (yavg>20):
            rm_sac[i:i+2] = True
            blinks.append((pss, nse))
            skip_next=True

    return array(blinks), saccades[~rm_sac,:]
