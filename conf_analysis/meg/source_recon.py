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
import numpy as np

from joblib import Memory
from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing

memory = Memory(cachedir=metadata.cachedir, verbose=0)


subjects_dir = '/home/nwilming/fs_subject_dir'
trans_dir = '/home/nwilming/conf_meg/trans'
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


def get_trans(subject, session):
    '''
    Return filename of transformation for a subject
    '''
    file_ident = 'S%i-SESS%i' % (subject, session)
    return join(trans_dir, file_ident + '-trans.fif')


@memory.cache
def get_info(subject, session):
    '''
    Return an info dict for a measurement from this subject.
    '''
    trans, fiducials, info = preprocessing.get_head_correct_info(
        subject, session)
    return info


@memory.cache
def get_bem(subject, head_model='three_layer', ico=4, type='mne'):

    if head_model == 'three_layer':
        conductivity = (0.3, 0.006, 0.3)  # for three layers
    else:
        conductivity = (0.3,)  # for single layer

    if type == 'mne':
        model = mne.make_bem_model(
            subject='S%02i' % subject,
            ico=ico,
            conductivity=conductivity,
            subjects_dir=subjects_dir)
    else:
        model = make_fieldtrip_bem_model(
            subject='S%02i' % subject,
            ico=None,
            conductivity=conductivity,
            subjects_dir=subjects_dir)
    return mne.make_bem_solution(model)


@memory.cache
def get_leadfield(subject, session, head_model='three_layer'):
    '''
    Compute leadfield with presets for this subject

    Parameters
    head_model : str, 'three_layer' or 'single_layer'
    '''
    from pymeg import source_reconstruction as pymegsr
    raw_filename = metadata.get_raw_filename(subject, session)
    epoch_filename = metadata.get_epoch_filename(
        subject, session, 0, 'stimulus', 'fif')
    trans = get_trans(subject, session)

    return pymegsr.get_leadfield('S%02i' % subject, raw_filename,
                                 epoch_filename, trans,
                                 conductivity=(0.3, 0.006, 0.3),
                                 bem_sub_path='bem_ft',
                                 njobs=4)
    '''

    src = get_source_space(subject)
    if head_model == 'three_layer':
        bem = get_bem(subject, head_model=head_model, type='fieldtrip')
    else:
        bem = get_bem(subject, head_model=head_model, type='mne')
    
    info = get_info(subject, session)

    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=2.5,
        n_jobs=4)
    return fwd, bem, fwd['src'], trans
    '''


def clear_cache():
    memory.clear()


def make_fieldtrip_bem_model(subject, ico=4, conductivity=(0.3, 0.006, 0.3),
                             subjects_dir=None, verbose=None):
    """Create a BEM model for a subject.

    Copied from MNE python, adapted to read surface from fieldtrip / spm
    segmentation.
    """
    import os.path as op
    from mne.io.constants import FIFF
    conductivity = np.array(conductivity, float)
    if conductivity.ndim != 1 or conductivity.size not in (1, 3):
        raise ValueError('conductivity must be 1D array-like with 1 or 3 '
                         'elements')
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir, raise_error=True)
    subject_dir = op.join(subjects_dir, subject)
    bem_dir = op.join(subject_dir, 'bem_ft')
    inner_skull = op.join(bem_dir, 'inner_skull.surf')
    outer_skull = op.join(bem_dir, 'outer_skull.surf')
    outer_skin = op.join(bem_dir, 'outer_skin.surf')
    surfaces = [inner_skull, outer_skull, outer_skin]
    ids = [FIFF.FIFFV_BEM_SURF_ID_BRAIN,
           FIFF.FIFFV_BEM_SURF_ID_SKULL,
           FIFF.FIFFV_BEM_SURF_ID_HEAD]
    print('Creating the BEM geometry...')
    if len(conductivity) == 1:
        surfaces = surfaces[:1]
        ids = ids[:1]
    surfaces = mne.bem._surfaces_to_bem(surfaces, ids, conductivity, ico)
    mne.bem._check_bem_size(surfaces)
    return surfaces


def add_volume_info(subject, surface, subjects_dir, volume='T1'):
    """Add volume info from MGZ volume
    """
    import os.path as op
    from mne.bem import _extract_volume_info
    from mne.surface import (read_surface, write_surface)
    subject_dir = op.join(subjects_dir, subject)
    mri_dir = op.join(subject_dir, 'mri')
    T1_mgz = op.join(mri_dir, volume + '.mgz')
    new_info = _extract_volume_info(T1_mgz)
    print(list(new_info.keys()))
    rr, tris, volume_info = read_surface(surface,
                                         read_metadata=True)

    # volume_info.update(new_info)  # replace volume info, 'head' stays
    print(list(volume_info.keys()))
    import numpy as np
    if 'head' not in list(volume_info.keys()):
        volume_info['head'] = np.array([2,  0, 20], dtype=np.int32)
    write_surface(surface, rr, tris, volume_info=volume_info)
