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



def clear_cache():
    memory.clear()
