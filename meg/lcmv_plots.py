import mne
import numpy as np
import matplotlib
import pylab as plt
from glob import glob
from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing, localizer, lcmv
from conf_analysis.meg import tfr_analysis as ta
from conf_analysis.meg import source_recon as sr
from joblib import Memory, Parallel, delayed

import pandas as pd

from itertools import product, izip


memory = Memory(cachedir=metadata.cachedir)



def overview_figure(subject, lowest_freq=None, F=None):
    '''
    Prepare data for an overview figure that shows source recon'ed activity.
    '''
    gs = matplotlib.gridspec.GridSpec(2 * 4, 10)
    stcs = get_stcs(subject, lowest_freq=lowest_freq, F=F)
    for col, view in enumerate(['cau', 'med']):
        for session, stc in enumerate(stcs):
            for j, hemi in enumerate(['lh', 'rh']):
                plt.subplot(gs[session * 2:session * 2 + 2, col * 2 + j])
                m = plot_stcs(stc, 'S%02i' % subject, hemi, vmin=2, vmax=12.5, view=view)
                plt.imshow(m, aspect='auto')
                plt.xticks([])
                plt.yticks([])

    offset = 3
    for session, sid in zip([0, 1, 2, 3], [0, 2, 4, 6]):
        n, l, r = preprocessing.get_head_loc(subject, session)
        plt.subplot(gs[sid:sid + 2, offset + 2])
        plt.plot(n)
        plt.plot(l)
        plt.plot(r)
        ticks = np.unique(np.around([np.mean(l), np.mean(n), np.mean(r)], 2))
        plt.yticks(ticks)
        plt.xticks([])
        # 1 Plot TFR for this participant and subject
        avg = lcmv.get_tfr(subject, session)

        chan, f, t = peak_channel(avg, 20)
        plt.subplot(gs[sid:sid + 2, offset + 3])
        plt.title('N=%i' % avg.nave)
        avg.plot_topomap(fmin=35, fmax=100, axes=plt.gca(), colorbar=False)
        plt.subplot(gs[sid:sid + 2, offset + 4:offset + 6])
        localizer.plot_tfr(avg,
                           np.array([76,  79,  80,  84,  89,  90, 205, 208, 209, 213, 218, 219, 268]))
        #avg.plot([chan], axes=plt.gca(), yscale='linear', colorbar=False)
        plt.xticks([0, 1])
        plt.yticks([20, 40, 60, 80, 100, 140])
        plt.xlabel('time')
        plt.ylabel('Hz')


def peak_channel(avg, fmin=10):
    id_f = fmin < avg.freqs
    chan, f, t = np.unravel_index(
        np.argmax(avg.data[:, id_f, :]),
        avg.data[:, id_f, :].shape)
    return chan, f + fmin, t


@memory.cache
def plot_stcs(stc, subject, hemi, view=['caud'], vmin=1, vmax=7.5):
    lim = {'kind': 'value', 'pos_lims': np.linspace(vmin, vmax, 3)}
    brain2 = stc.plot(subject=subject,
                      subjects_dir=sr.subjects_dir,
                      hemi=hemi,
                      views=view,
                      clim=lim, size=(750, 750),
                      )
    m = brain2.screenshot()
    return m


def get_stcs(subject, lowest_freq=None, F=None):
    stcs = []
    for session in [0, 1, 2, 3]:
        s = get(subject, session, lowest_freq=lowest_freq, F=F)
        s = zscore(s)
        s = avg(s)
        stcs.append(s)
    return stcs


def get(subject, session, lowest_freq=None, F=None):
    if F is None:
        files = glob('/home/nwilming/conf_meg/source_recon/SR_S%i_SESS%i*' %
                     (subject, session))
    else:
        search = '/home/nwilming/conf_meg/source_recon/SR_S%i_SESS%i_lF%i_F%i*' %\
                     (subject, session, lowest_freq, F)
        print(search)
        files = glob(search)
        print(files)
    stcs = reduce(lambda x, y: x + y,
                  [mne.read_source_estimate(s) for s in files])
    if stcs.times[0] <= -1.5:
        times = stcs.times + 0.75
        stcs = mne.SourceEstimate(
            stcs.data, stcs.vertices, times[0], np.diff(times)[0])
    return stcs


def zscore(stc, baseline=(-0.25, 0)):
    data = stc.data
    idbase = (baseline[0] < stc.times) & (stc.times < baseline[1])
    m = data[:, idbase].mean(1)[:, np.newaxis]
    s = data[:, idbase].std(1)[:, np.newaxis]
    data = (data - m) / s
    stc.data = data
    return stc


def avg(stc):
    id_t = (0.1 < stc.times) & (stc.times < .5)
    d = stc.data[:, id_t].mean(1)[:, np.newaxis]
    stc = mne.SourceEstimate(d, stc.vertices, tmin=0, tstep=1)
    return stc
