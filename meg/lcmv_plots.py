import mne
import numpy as np
import matplotlib
import pylab as plt

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing, localizer
from conf_analysis.meg import tfr_analysis as ta
from conf_analysis.meg import source_recon as sr
from joblib import Memory, Parallel, delayed

import pandas as pd

from itertools import product, izip


memory = Memory(cachedir=metadata.cachedir)


def overview_figure(avg):
    '''
    Prepare data for an overview figure that shows source recon'ed activity.
    '''
    # 1 Plot TFR for this participant and subject
    # ep = ta.get_sub_sess_object(subject, session, (10, 150), None, -0.4, 1.1)
    # avg = ep.average()
    # del ep
    # avg = avg.apply_baseline((-0.2, 0), mode='zscore')
    chan, f, t = peak_channel(avg, 20)
    plt.subplot(1, 2, 1)
    avg.plot_topomap(fmin=35, fmax=100, axes=plt.gca())
    plt.subplot(1, 2, 2)
    avg.plot([chan], axes=plt.gca())


def plot_stcs(stcs, view=['med'], vmin=1, vmax=7.5):
    gs = matplotlib.gridspec.GridSpec(len(stcs), 2)
    lim = {'kind': 'value', 'pos_lims': np.linspace(vmin, vmax, 3)}
    for i, s in enumerate(stcs):
        brain2 = s.plot(subject='S02',
                        subjects_dir='/Users/nwilming/u/freesurfer_subjects/',
                        hemi='lh',
                        views=view, clim=lim, size=(500, 250))
        m = brain2.screenshot_single()
        plt.subplot(gs[i, 0])
        plt.imshow(m)
        plt.xticks([])
        plt.yticks([])
    for i, s in enumerate(stcs):
        brain2 = s.plot(subject='S02',
                        subjects_dir='/Users/nwilming/u/freesurfer_subjects/',
                        hemi='rh',
                        views=view, clim=lim, size=(500, 250))
        m = brain2.screenshot_single()
        plt.subplot(gs[i, 1])
        plt.imshow(m)
        plt.xticks([])
        plt.yticks([])


def get_stcs(subject):
    stcs = []
    for session in [0, 1, 2, 3]:
        s = get(subject, session)
        s = zscore(s)
        s = avg(s)
        stcs.append(s)
    return s


def get(subject, session):
    files = glob('/Users/nwilming/Desktop/source_recon/SR_S%i_SESS%i*' %
                 (subject, session))
    stcs = reduce(lambda x, y: x + y,
                  [mne.read_source_estimate(s) for s in files])
    if stcs.times[0] <= -1.5:
        times = stcs.times + 0.75
        stcs = mne.SourceEstimate(
            stcs.data, stcs.vertices, times[0], diff(times)[0])
    return stcs


def zscore(stc, baseline=(-0.25, 0)):
    data = stc.data
    idbase = (baseline[0] < stc.times) & (stc.times < baseline[1])
    m = data[:, idbase].mean(1)[:, np.newaxis]
    s = data[:, idbase].std(1)[:, np.newaxis]
    data = (data - m) / s
    stc.data = data


 def avg(stc):
    id_t = (0.1<stc.times) & (stc.times <.5)
    d = stc.data[:, id_t].mean(1)[:, newaxis]
    stc = mne.SourceEstimate(d, stc.vertices, tmin=0, tstep=1)
    return stc
    return stc
