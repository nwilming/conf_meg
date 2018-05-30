from __future__ import division
from __future__ import print_function

import mne
import numpy as np
import pandas as pd

from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing
from conf_analysis.meg import source_recon as sr

from joblib import Memory

from mne import compute_covariance
from mne.beamformer import lcmv_epochs

from os import makedirs
from os.path import join
from pymeg import tfr, lcmv as pymeglcmv


from itertools import izip


memory = Memory(cachedir=metadata.cachedir)


def submit():
    from pymeg import parallel
    for subject in range(1, 16):
        for session in range(4):
            for epoch in ['stimulus', 'response']:
                parallel.pmap(
                    run_and_save, [(subject, session, epoch)],
                    walltime=10, memory=70, nodes='1:ppn=4',
                    name='SR' + str(subject) + '_' + str(session) + epoch,
                    ssh_to=None)


@memory.cache
def get_cov(epochs, tmin=0, tmax=1):
    return compute_covariance(epochs, tmin=tmin, tmax=tmax, method='shrunk')


@memory.cache
def get_noise_cov(epochs):
    return compute_covariance(epochs, tmin=-0.5, tmax=0, method='shrunk')


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

    estimators = (get_broadband_estimator(),
                  get_highF_estimator(),
                  get_lowF_estimator())
    accum = AccumSR(srfilename(subject, session, BEM, epoch_type),
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
        data_cov = get_cov(epochs, tmin=0, tmax=1.15)

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
        data_cov = get_cov(epochs, tmin=0, tmax=1.15)
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
    ''' This is the old style:
    source_epochs = do_epochs(epochs, meta, forward, source,
                              noise_cov,
                              data_cov,
                              labels,
                              func=func,
                              accumulator=accumulator)
    '''
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




def do_epochs(epochs, meta, forward, source, noise_cov, data_cov, labels,
              func=None, accumulator=None):
    '''    
    Perform SR for a set of epochs.

    func can apply a function to the source-reconstructed data, for example
    to perform a time-frqeuncy decomposition. The behavior of do_epochs depends
    on what this function returns. If it returns a 2D matrix, it will be
    interpreted as source_locations*num_F*time array.

    func itself should be a list of tuples: ('identifier', identifier values, function).
    This allows to label the transformed data appropriately.
    '''
    
    results = []
    if labels is None:
        labels = []
    if meta is None:
        index = np.arange(epochs._data.shape[0])
    else:
        index = epochs.events[:, 2]
    for trial, epoch in izip(index,
                             lcmv_epochs(epochs, forward, noise_cov, data_cov,
                                         reg=0.05,
                                         pick_ori='max-power',
                                         return_generator=True)):
        if func is None:
            srcepoch = extract_labels_from_trial(
                epoch, labels, int(trial), source)
            results.append(srcepoch)
            if accumulator is not None:
                accumulator.update(epoch)
            del epoch
        else:
            for keyname, values, function in func:
                print('Running', keyname, 'on trial', int(trial))
                transformed = function(epoch.data)

                tstep = epoch.tstep / \
                    (float(transformed.shape[2]) / len(epoch.times))

                for value, row in zip(values, np.arange(transformed.shape[1])):

                    new_epoch = mne.SourceEstimate(transformed[:, row, :],
                                                   vertices=epoch.vertices,
                                                   tmin=epoch.tmin,
                                                   tstep=tstep,
                                                   subject=epoch.subject)

                    srcepoch = extract_labels_from_trial(
                        new_epoch, labels, int(trial), source)
                    srcepoch['est_val'] = value
                    srcepoch['est_key'] = keyname
                    results.append(srcepoch)
                    if accumulator is not None:
                        accumulator.update(keyname, value, new_epoch)
                    del new_epoch
                del transformed
            del epoch

    if len(labels) > 0:
        results = pd.concat([to_df(r) for r in results])
    else:
        results = None
    return results


def extract_labels_from_trial(epoch, labels, trial, source):
    srcepoch = {'time': epoch.times, 'trial': trial}
    for label in labels:
        try:
            pca = epoch.extract_label_time_course(
                label, source, mode='mean')
        except ValueError:
            pass
            # print('Source space contains no vertices for', label)
        srcepoch[label.name] = pca
    return srcepoch


class AccumSR(object):
    '''
    Accumulate SRs and compute an average.
    '''

    def __init__(self, filename, keyname, value):
        self.stc = None
        self.N = 0
        self.filename = filename
        self.keyname = keyname
        self.value = value

    def update(self, keyname, value, stc):
        if (self.keyname == keyname) and (self.value == value):
            if self.stc is None:
                self.stc = stc.copy()
            else:
                self.stc += stc
            self.N += 1

    def save_averaged_sr(self):
        stcs = self.stc.copy()
        idbase = (-.5 < stcs.times) & (stcs.times < 0)
        m = stcs.data[:, idbase].mean(1)[:, np.newaxis]
        s = stcs.data[:, idbase].std(1)[:, np.newaxis]
        stcs.data = (stcs.data - m) / s
        stcs.save(self.filename)
        return stcs


def get_highF_estimator(sf=600, decim=10):
    fois = np.arange(10, 151, 5)
    cycles = 0.1 * fois
    tb = 2
    return ('F', fois, get_power_estimator(fois, cycles, tb, sf=sf, decim=decim))


def get_lowF_estimator(sf=600, decim=10):
    fois = np.arange(1, 21, 2)
    cycles = 0.25 * fois
    tb = 2
    return ('LF', fois, get_power_estimator(fois, cycles, tb, sf=sf, decim=decim))


def get_broadband_estimator():
    return ('BB', [-1], lambda x: x[:, np.newaxis, :])


def get_power_estimator(F, cycles, time_bandwidth, sf=600., decim=1):
    '''
    Estimate power from source reconstruction

    This will return a num_trials*num_F*time array
    '''
    import functools

    def foo(x, sf=600.,
            foi=None,
            cycles=None,
            time_bandwidth=None,
            n_jobs=None, decim=decim):
        x = x[np.newaxis, :, :]
        x = tfr.array_tfr(x,
                          sf=sf,
                          foi=foi,
                          cycles=cycles,
                          time_bandwidth=time_bandwidth,
                          n_jobs=4, decim=decim)
        return x.squeeze()

    return functools.partial(foo, sf=sf,
                             foi=F,
                             cycles=cycles,
                             time_bandwidth=time_bandwidth,
                             n_jobs=4, decim=decim)


def to_df(r):
    length = len(r['time'])
    p = {}

    for key in r.keys():
        try:
            p[key] = r[key].ravel()
            if len(p[key]) == 1:
                p[key] = [r[key]] * length
        except AttributeError:
            p[key] = [r[key]] * length
    return pd.DataFrame(p)
