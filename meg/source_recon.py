'''
Compute source reconstruction for all subjects

Succesfull source reconstruction depends on a few things:

fMRI
    1. Recon-all MRI for each subject
    2. mne watershed_bem for each subject
    3. mne make_scalp_surfaces for all subjects
    4. Coreg the Wang & Kastner atlas to each subject using the scripts
       in require/apply_occ_wang/bin (apply_template and to_label)
MEG:
    5. A piece of sample data for each subject (in fif format)
    6. Create a coregistration for each subjects MRI with mne coreg
    7. Create a source space
    8. Create a bem model
    9. Create a leadfield
   10. Compute a noise and data cross spectral density matrix
       -> Use a data CSD that spans the entire signal duration.
   11. Run DICS to project epochs into source space
   12. Extract time course in labels from Atlas.

'''
from os.path import join
import mne
import pandas as pd

from joblib import Memory
from conf_analysis.behavior import metadata
memory = Memory(cachedir=metadata.cachedir, verbose=0)


subjects_dir = '/home/nwilming/fs_subject_dir'
plot_dir = '/home/nwilming/conf_analysis/plots/source'


def check_bems():
    '''
    Create a plot of all BEM segmentations
    '''
    for sub in range(1, 16):
        fig = mne.viz.plot_bem(subject='S%02i' % sub,
                               subjects_dir=subjects_dir,
                               brain_surfaces='white',
                               orientation='coronal')
        fig.savefig(join(plot_dir, 'bem', 'check_S%02i.png' % sub))


@memory.cache
def get_source_space(subject):
    '''
    Return source space.
    Abstract this here to have it unique for everyone.
    '''
    return mne.setup_source_space('S%02i' % subject, spacing='oct6',
                                  subjects_dir=subjects_dir,
                                  add_dist=False)


def get_trans(subject):
    '''
    Return filename of transformation for a subject
    '''
    subject = 'S%02i' % subject
    return join(subjects_dir, subject + '-trans.fif')


@memory.cache
def get_info(subject):
    '''
    Return an info dict for a measurement from this subject.
    '''
    subject = 'S%02i' % subject
    filename = join(subjects_dir, subject + '-raw.fif')
    return mne.io.read_raw_fif(filename).info


@memory.cache
def get_bem(subject):
    conductivity = (0.3,)  # for single layer
    # conductivity = (0.3, 0.006, 0.3)  # for three layers
    model = mne.make_bem_model(
        subject='S%02i' % subject,
        ico=4,
        conductivity=conductivity,
        subjects_dir=subjects_dir)
    return mne.make_bem_solution(model)


@memory.cache
def get_leadfield(subject):
    '''
    Compute leadfield with presets for this subject
    '''
    src = get_source_space(subject)
    bem = get_bem(subject)
    trans = get_trans(subject)

    fwd = mne.make_forward_solution(
        get_info(subject),
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=2)
    #fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True,
    #                                         force_fixed=True)
    return fwd, bem, fwd['src'], trans


@memory.cache
def get_labels(subject):
    import glob
    subject = 'S%02i' % subject
    subject_dir = join(subjects_dir, subject)
    labels = glob.glob(join(subject_dir, 'label', '*wang2015atlas*'))
    return [mne.read_label(label, subject) for label in labels]


def get_eccen_labels(subject):
    import glob
    subject = 'S%02i' % subject
    subject_dir = join(subjects_dir, subject)
    labels = glob.glob(join(subject_dir, 'label', '*eccen11*'))
    return [mne.read_label(label, subject) for label in labels]


def save_source_power(data, subject, filename):
    import numpy as np
    transform = lambda x: dict((d, np.asarray(k).ravel() if type(
        k) == np.ndarray else k) for d, k in x.iteritems())
    df = pd.concat([pd.DataFrame(transform(x)) for x in data])
    df.loc[:, 'snum'] = subject
    df.to_hdf(filename, 'srcpow')


@memory.cache
def get_stimulus_csd(epochs, noise_t=(0.25, 0.75), data_t=(0.75, 1.75),
                     freq=(45, 65)):
    from mne.time_frequency import csd_epochs
    # Construct filter
    noise_csd = csd_epochs(epochs,
                           mode='multitaper', tmin=noise_t[0], tmax=noise_t[1],
                           fmin=freq[0], fmax=freq[1], fsum=True)
    data_csd = csd_epochs(epochs,
                          mode='multitaper', tmin=data_t[0], tmax=data_t[1],
                          fmin=freq[0], fmax=freq[1], fsum=True)
    return noise_csd, data_csd


def foo(source_epoch, labels, i, events):
    d = []
    for label in labels:
        data = {}
        hemi = 0 if 'lh' == label.hemi else 1
        l_data = source_epoch.in_label(label)
        data['hemi'] = hemi
        if hemi == 0:
            for vertex, vert_data in zip(l_data.vertices[0],
                                         l_data.lh_data):
                data[vertex] = vert_data
        elif hemi == 1:
            for vertex, vert_data in zip(l_data.vertices[1],
                                         l_data.rh_data):
                data[vertex] = vert_data

        data['trial'] = events[i, 2]
        data['time'] = source_epoch.times
        d.append(pd.DataFrame(data))
    return pd.concat(d)


def localize_stim_power(subject):
    from conf_analysis.meg import preprocessing
    epochs, meta = preprocessing.get_epochs_for_subject(subject, 'stimulus')
    epochs = epochs.apply_baseline((0.5, 0.75))
    forward, bem, source, trans = get_leadfield(subject)
    stc = localize_power_changes(epochs, forward)[0]
    sub = 'S%02i' % subject
    stc.save('/home/nwilming/conf_data/%s-stim-power-localized.source' % sub)
    return stc


def localize_power_changes(epochs, forward, tmin=0.75, tmax=1.75, tstep=0.05):
    from mne.time_frequency import csd_epochs
    from mne.beamformer import tf_dics
    # Setting frequency bins as in Dalal et al. 2008
    freq_bins = [(45, 65)]  # Hz
    win_lengths = [0.1]  # s
    # Then set FFTs length for each frequency range.
    # Should be a power of 2 to be faster.
    n_ffts = [128]

    # Subtract evoked response prior to computation?
    subtract_evoked = False

    # Calculating noise cross-spectral density from empty room noise for each
    # frequency bin and the corresponding time window length. To calculate
    # noise from the baseline period in the data, change epochs_noise to epochs
    noise_csds = []
    for freq_bin, win_length, n_fft in zip(freq_bins, win_lengths, n_ffts):
        noise_csd = csd_epochs(epochs, mode='fourier',
                               fmin=freq_bin[0], fmax=freq_bin[1],
                               fsum=True, tmin=tmin - win_length, tmax=tmax,
                               n_fft=n_fft)
        noise_csds.append(noise_csd)

    # Computing DICS solutions for time-frequency windows in a label in source
    # space for faster computation, use label=None for full solution
    stcs = tf_dics(epochs, forward, noise_csds, tmin, tmax, tstep, win_lengths,
                   freq_bins=freq_bins, subtract_evoked=subtract_evoked,
                   n_ffts=n_ffts, reg=0.001)
    return stcs


def plot_montage(stc, subject, row=0, gs=None):
    '''
    Plot a montage of max power and brain areas.
    '''
    import seaborn as sns
    import numpy as np
    import pylab as plt
    import matplotlib
    brain = stc.plot(subject,
                     subjects_dir='/Users/nwilming/cluster_home/fs_subject_dir',
                     background=(1., 1., 1.))
    view = dict(azimuth=-70, elevation=100)
    brain.show_view(view=view)
    labels = ['V1d', 'V1v', 'V2d', 'V2v', 'V3d', 'V3v', 'hV4', 'LO1', 'LO2']
    colors = sns.color_palette(n_colors=len(labels))
    for i, label in enumerate(labels):
        brain.add_label('wang2015atlas.' + label,
                        color=colors[i], borders=True)

    o = []
    for t in np.linspace(0.75, 1.7, 11):
        brain.set_time(t)
        o.append(brain.save_montage(None, order=[view]))

    if gs is None:
        gs = matplotlib.gridspec.GridSpec(1, len(o))
    for i, img in enumerate(o):
        plt.subplot(gs[row, i])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    return o


def clear_cache():
    memory.clear()
