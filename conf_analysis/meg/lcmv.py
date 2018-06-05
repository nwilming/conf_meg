


import mne
import numpy as np

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing
from conf_analysis.meg import source_recon as sr

from joblib import Memory

from os import makedirs
from os.path import join
from pymeg import lcmv as pymeglcmv


memory = Memory(cachedir=metadata.cachedir)


def submit():
    from pymeg import parallel
    for subject in range(1, 16):
        for session in range(4):
            for epoch in ['stimulus', 'response']:
                parallel.pmap(
                    run_and_save, [(subject, session, epoch)],
                    walltime=10, memory=10, nodes='1:ppn=1',
                    name='SR' + str(subject) + '_' + str(session) + epoch,
                    ssh_to=None)


def run_and_save(subject, session, epoch_type='stimulus', BEM='three_layer', debug=False):
    mne.set_log_level('WARNING')

    def lcmvfilename(subject, session, BEM, epoch_type):
        path = '/home/nwilming/conf_meg/sr_labeled/'
        try:
            makedirs(path)
        except:
            pass
        filename = 'S%i-SESS%i-%s-lcmv.hdf' % (
            subject, session, epoch_type)
        return join(path, filename)

    def srfilename(subject, session, BEM, epoch_type):
        path = '/home/nwilming/conf_meg/sr_averages/'
        try:
            makedirs(path)
        except:
            pass
        filename = 'S%i-SESS%i-%s.stc' % (
            subject, session, epoch_type)
        return join(path, filename)

    estimators = (pymeglcmv.get_broadband_estimator(),
                  pymeglcmv.get_highF_estimator(),
                  pymeglcmv.get_lowF_estimator())
    accum = pymeglcmv.AccumSR(srfilename(subject, session, BEM, epoch_type),
                              'F', 55)
    epochs = extract(subject, session, func=estimators,
                     accumulator=accum, debug=debug, epoch_type=epoch_type)
    epochs.to_hdf(lcmvfilename(subject, session, BEM, epoch_type), 'epochs')
    accum.save_averaged_sr()


def extract(subject, session,
            func=None, accumulator=None,
            BEM='three_layer', debug=False, epoch_type='stimulus'):

    if epoch_type == 'stimulus':
        epochs, meta = preprocessing.get_epochs_for_subject(subject,
                                                            'stimulus',
                                                            sessions=session)
        epochs = epochs.pick_channels(
            [x for x in epochs.ch_names if x.startswith('M')])
        id_time = (-0.25 <= epochs.times) & (epochs.times <= 0)
        means = epochs._data[:, :, id_time].mean(-1)
        epochs._data -= means[:, :, np.newaxis]
        data_cov = pymeglcmv.get_cov(epochs, tmin=0, tmax=1.15)

    elif epoch_type == 'response':
        epochs, meta = preprocessing.get_epochs_for_subject(subject,
                                                            'stimulus',
                                                            sessions=session)
        epochs = epochs.pick_channels(
            [x for x in epochs.ch_names if x.startswith('M')])
        response, meta = preprocessing.get_epochs_for_subject(subject,
                                                              'response',
                                                              sessions=session)
        response = response.pick_channels(
            [x for x in response.ch_names if x.startswith('M')])
        # Find trials that are present in both time periods
        overlap = list(
            set(epochs.events[:, 2]).intersection(
                set(response.events[:, 2])))
        epochs = epochs[[str(l) for l in overlap]]
        response = response[[str(l) for l in overlap]]
        id_time = (-0.25 <= epochs.times) & (epochs.times <= 0)
        means = epochs._data[:, :, id_time].mean(-1)
        epochs._data -= means[:, :, np.newaxis]

        response._data = (
            response._data - means[:, :, np.newaxis])

        # Now also baseline stimulus period
        epochs, meta = preprocessing.get_epochs_for_subject(subject,
                                                            'stimulus',
                                                            sessions=session)
        epochs = epochs.pick_channels(
            [x for x in epochs.ch_names if x.startswith('M')])
        id_time = (-0.25 <= epochs.times) & (epochs.times <= 0)
        means = epochs._data[:, :, id_time].mean(-1)
        epochs._data -= means[:, :, np.newaxis]
        data_cov = pymeglcmv.get_cov(epochs, tmin=0, tmax=1.15)
        epochs = response
    else:
        raise RuntimeError('Did not recognize epoch')

    forward, bem, source, trans = sr.get_leadfield(subject, session, BEM)
    labels = sr.get_labels(subject)
    # labels = [l for l in labels if 'V' in l.name]
    if debug:
        # Select only 2 trials to make debuggin easier
        trials = meta.index.values[:2]
        epochs = epochs[[str(l) for l in trials]]
        meta = meta.loc[trials, :]

    source_epochs = pymeglcmv.reconstruct(
        epochs=epochs,
        forward=forward,
        source=source,
        noise_cov=None,
        data_cov=data_cov,
        labels=labels,
        func=func,
        accumulator=accumulator,
        first_all_vertices=False)
    return source_epochs
