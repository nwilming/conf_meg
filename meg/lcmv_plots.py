import mne
import numpy as np
import matplotlib
import pylab as plt
from glob import glob
from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing, localizer, lcmv, srplots
from conf_analysis.meg import tfr_analysis as ta
from conf_analysis.meg import source_recon as sr
from joblib import Memory, Parallel, delayed

import pandas as pd

from itertools import product, izip


memory = Memory(cachedir=metadata.cachedir)


def overview_figure(subject):
    '''
    Prepare data for an overview figure that shows source recon'ed activity.
    '''
    plt.figure(figsize=(15, 15))
    gs = matplotlib.gridspec.GridSpec(2 * 4, 6)
    freqs, _, stcfiles = srplots.get_freq_tuning(subject)
    stcs = get_stcs(stcfiles)
    for col, view in enumerate(['cau', 'med', 'lat']):
        for session, stc in enumerate(stcs):
            for j, hemi in enumerate(['lh', 'rh']):
                plt.subplot(gs[session * 2:session * 2 + 2, col * 2 + j])
                m = plot_stcs(stc, 'S%02i' % subject, hemi,
                              vmin=2, vmax=12.5, view=view)
                plt.imshow(m)
                plt.xticks([])
                plt.yticks([])
    plt.savefig('/home/nwilming/sub_%i_stc_overview.png'%subject, dpi=600)
    plt.figure(figsize=(15, 10))
    gs = matplotlib.gridspec.GridSpec(2 * 4, 7)
    offset = -2
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
        avg = lcmv.load_tfr(subject, session)
        channels = lcmv.select_channels_by_gamma(avg, n=3)
        chan, f, t = peak_channel(avg, 20)
        plt.subplot(gs[sid:sid + 2, offset + 3])
        plt.title('N=%i' % avg.nave)
        avg.plot_topomap(fmin=35, fmax=100, axes=plt.gca(), colorbar=False)
        plt.subplot(gs[sid:sid + 2, offset + 5:offset + 7])
        localizer.plot_tfr(avg, channels)
        #avg.plot([chan], axes=plt.gca(), yscale='linear', colorbar=False)
        plt.xticks([0, 1])
        plt.yticks([20, 40, 60, 80, 100, 140])
        plt.xlabel('time')
        plt.ylabel('Hz')

        plt.subplot(gs[sid:sid + 2, offset + 8])
        power, meta = srplots.get_power(
            subject, session=session, decim=3, tuning=1)
        sa = srplots.sample_aligned_power(
            power, meta, 'V1dlh', baseline=(-0.2, 0))
        srplots.plot_sample_aligned_power(
            sa, edges=[0, .4, .6, 1], ax=plt.gca())
        plt.xticks([0, 0.1, 0.2, 0.3])
        yd = np.abs(plt.ylim()).max()/2
        plt.yticks([-yd, 0, yd])
        plt.legend([])
    plt.legend()
    plt.savefig('/home/nwilming/sub_%i_stats_overview.png'%subject, dpi=600)

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


def get_stcs(filenames):
    stcs = []
    for session in [0, 1, 2, 3]:
        s = get(filenames[session])
        s = zscore(s)
        s = avg(s)
        stcs.append(s)
    return stcs


def get(files):
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
