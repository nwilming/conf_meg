
import numpy as np
import mne
from conf_analysis.meg import source_recon as sr
from conf_analysis.behavior import metadata

from copy import deepcopy

from scipy import linalg

from mne.beamformer._lcmv import (_prepare_beamformer_input, _setup_picks,
                                  _reg_pinv)
from mne.time_frequency import csd_epochs as mne_csd_epochs
from mne.externals import six
from joblib import Memory
from joblib import Parallel, delayed

import time as clock


memory = Memory(cachedir=metadata.cachedir)


@memory.cache
def get_tfr(subject, n_blocks=None):
    from glob import glob
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


@memory.cache
def make_dics_filter(epochs, freqs, f, times, f_smooth, t_smooth, subject,
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
            delayed(one_csd_and_filter)(i, epochs, forward, noise_csd, time,
                                        fmin, fmax, t_smooth)
            for i, time in enumerate(times[idx]))

        print 'Done with CSDds'
        #filters = {}
        # filters = parallel(
        #    delayed(one_filter)(time, epochs.info, forward, noise_csd, dcsd)
        #    for time, dcsd in data_csds)

    #filters = dict((i, filter['weights']) for i, filter in filters)
    return f, filters


def one_csd_and_filter(i, epochs, forward, noise_csd, time, fmin, fmax,
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
    dics_filter =  one_filter(t, info, forward, noise_csd, data_csd)
    print i, 'CSD + Filter took', np.around(clock.time() - start, 3)
    return dics_filter

def one_filter(time, info, forward, noise_csd, data_csd):
    # data_csd = csd_epochs(epochs, 'multitaper', fmin, fmax,
    #                      fsum=True,
    #                      tmin=time - t_smooth,
    #                      tmax=time + t_smooth)
    dics_filter = make_dics(info, forward, noise_csd, data_csd)
    return time, dics_filter


@memory.cache
def get_noise_csd(epochs, fmin, fmax, t_smooth):
    return mne_csd_epochs(epochs, 'multitaper', fmin, fmax,
                         fsum=True, tmin=0.75 - 2 * t_smooth,
                         tmax=0.75)


def apply_dics_filter(filters, f, tfr, filename, n_jobs=12):
    '''
    Apply beamformer to TFR data
    '''
    id_freq = np.argmin(np.abs(tfr.freqs - f))
    k0 = filters[0].keys()
    forward = filters[0][k0]['forward']
    sp_shape = (forward['nsource'], tfr.data.shape[0], tfr.data.shape[3])
    source_pow = np.memmap(filename, dtype='float32', mode='w+',
                           shape=sp_shape)

    # for i, (t, A) in enumerate(filters.iteritems()):
    #    A = A['weights']

    #    for epoch in range(tfr.data.shape[0]):
    #       Xsensor = tfr.data[epoch, :, id_freq,  i]  # 269; Avoid copy
    #        Xsource = np.dot(A, Xsensor)  # 8k = (8k x 269) * (269,)
    #        source_pow[:, epoch, i] = Xsource * np.conj(Xsource)
    print 'Applying in parallel'
    Parallel(n_jobs=n_jobs)(
        delayed(apply_one_filter)(
            source_pow, tfr.data[epoch, :, id_freq],
            epoch,
            i,
            A['weights'])
        for i, epoch in enumerate(range(tfr.data.shape[0]))
        for (t, A) in filters.iteritems())

    return source_pow


def apply_one_filter(source_pow, Xsensor, epoch, i, A):
    # Xsensor = tfrdata[epoch, :, id_freq,  i]  # 269; Avoid copy
    Xsource = np.dot(A, Xsensor)  # 8k = (8k x 269) * (269,)
    source_pow[:, epoch, i] = Xsource * np.conj(Xsource)


def make_dics(info, forward, noise_csd, data_csd, reg=0.05, label=None,
              pick_ori=None, real_filter=False, verbose=None):
    """
    From: britta-wstnr
    https://github.com/britta-wstnr/mne-python/blob/5ccb0b7a4803aea4e1f95aac9f932f75ae17b1d6/mne/beamformer/_dics.py

    Compute Dynamic Imaging of Coherent Sources (DICS) spatial filter.
    .. note:: Fixed orientation forward operators with ``real_filter=False``
              will result in complex time courses, in which case absolute
              values will be returned.
    .. note:: This implementation has not been heavily tested so please
              report any issues or suggestions.
    Parameters
    ----------
    info : dict
        The measurement info to specify the channels to include.
        Bad channels in info['bads'] are not used.
    forward : dict
        Forward operator.
    noise_csd : instance of CrossSpectralDensity
        The noise cross-spectral density.
    data_csd : instance of CrossSpectralDensity
        The data cross-spectral density.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    real_filter : bool
        If True, take only the real part of the cross-spectral-density matrices
        to compute real filters as in [2]_. Default is False.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    Returns
    -------
    filters : dict
        Dictionary containing filter weights from DICS beamformer.
        Contains the following keys:
            'weights' : {array}
                The filter weights of the beamformer.
            'data_csd' : {instance of CrossSpectralDensity}
                The data cross-spectral density matrix used to compute the
                beamformer.
            'noise_csd' : {instance of CrossSpectralDensity}
                The noise cross-spectral density matrix used to compute the
                beamformer.
            'pick_ori' : {None | 'normal'}
                Orientation selection used in filter computation.
            'ch_names' : {list}
                Channels used to compute the beamformer.
            'proj' : {array}
                Projections used to compute the beamformer.
            'vertices' : {list}
                Vertices for which the filter weights were computed.
            'is_free_ori' : {bool}
                If True, the filter was computed with free source orientation.
            'src' : {instance of SourceSpaces}
                Source space information.
    See Also
    --------
    apply_dics, dics, dics_epochs
    Notes
    -----
    For more information about ``real_filter``, see the
    `supplemental information <http://www.cell.com/cms/attachment/616681/4982593/mmc1.pdf>`_
    from [2]_.
    References
    ----------
    .. [1] Gross et al. Dynamic imaging of coherent sources: Studying neural
           interactions in the human brain. PNAS (2001) vol. 98 (2) pp. 694-699
    .. [2] Hipp JF, Engel AK, Siegel M (2011) Oscillatory Synchronization
           in Large-Scale Cortical Networks Predicts Perception.
           Neuron 69:387-396.
    """  # noqa: E501
    picks = _setup_picks(info, forward)

    is_free_ori, ch_names, proj, vertno, G =\
        _prepare_beamformer_input(info, forward, label, picks, pick_ori)

    Cm = data_csd.data.copy()

    # Take real part of Cm to compute real filters
    if real_filter:
        Cm = Cm.real

    # Tikhonov regularization using reg parameter to control for
    # trade-off between spatial resolution and noise sensitivity
    # eq. 25 in Gross and Ioannides, 1999 Phys. Med. Biol. 44 2081
    Cm_inv, _ = _reg_pinv(Cm, reg)
    del Cm

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient

    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        Ck = np.dot(Wk, Gk)

        # TODO: max-power is not implemented yet, however DICS does employ
        # orientation picking when one eigen value is much larger than the
        # other

        if is_free_ori:
            # Free source orientation
            Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
        else:
            # Fixed source orientation
            Wk /= Ck

        # Noise normalization
        noise_norm = np.dot(np.dot(Wk.conj(), noise_csd.data), Wk.T)
        noise_norm = np.abs(noise_norm).trace()
        Wk /= np.sqrt(noise_norm)

    # Pick source orientation normal to cortical surface
    if pick_ori == 'normal':
        W = W[2::3]
        is_free_ori = False

    filters = dict(weights=W, data_csd=data_csd, noise_csd=noise_csd,
                   pick_ori=pick_ori, ch_names=ch_names, proj=proj,
                   vertices=vertno, is_free_ori=is_free_ori,
                   src=deepcopy(forward['src']), forward=forward)

    return filters


def csd_epochs(epochs, times, sfreq, ch_names, mode='multitaper', fmin=0, fmax=np.inf,
               tmin=None, tmax=None, n_fft=None,
               mt_bandwidth=None, mt_adaptive=False, mt_low_bias=True,
               projs=None, verbose=None, parallel=None):
    """Estimate cross-spectral density from epochs.
    Note: Baseline correction should be used when creating the Epochs.
          Otherwise the computed cross-spectral density will be inaccurate.
    Note: Results are scaled by sampling frequency for compatibility with
          Matlab.
    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    mode : str
        Spectrum estimation mode can be either: 'multitaper' or 'fourier'.
    fmin : float
        Minimum frequency of interest.
    fmax : float | np.inf
        Maximum frequency of interest.
    fsum : bool
        Sum CSD values for the frequencies of interest. Summing is performed
        instead of averaging so that accumulated power is comparable to power
        in the time domain. If True, a single CSD matrix will be returned. If
        False, the output will be a list of CSD matrices.
    tmin : float | None
        Minimum time instant to consider. If None start at first sample.
    tmax : float | None
        Maximum time instant to consider. If None end at last sample.
    n_fft : int | None
        Length of the FFT. If None the exact number of samples between tmin and
        tmax will be used.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'multitaper' mode.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
        Only used in 'multitaper' mode.
    mt_low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth. Only used in 'multitaper' mode.
    projs : list of Projection | None
        List of projectors to use in CSD calculation, or None to indicate that
        the projectors from the epochs should be inherited.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    Returns
    -------
    csd : instance of CrossSpectralDensity
        The computed cross-spectral density.
    """
    # Portions of this code adapted from mne/connectivity/spectral.py

    # Check correctness of input data and parameters
    from mne.time_frequency.multitaper import (dpss_windows, _mt_spectra,
                                               _csd_from_mt)

    if fmax < fmin:
        raise ValueError('fmax must be larger than fmin')
    tstep = times[1] - times[0]
    if tmin is not None and tmin < times[0] - tstep:
        raise ValueError('tmin should be larger than the smallest data time '
                         'point')
    if tmax is not None and tmax > times[-1] + tstep:
        raise ValueError('tmax should be smaller than the largest data time '
                         'point')
    if tmax is not None and tmin is not None:
        if tmax < tmin:
            raise ValueError('tmax must be larger than tmin')

    # Preparing time window slice
    tstart, tend = None, None
    if tmin is not None:
        tstart = np.where(times >= tmin)[0][0]
    if tmax is not None:
        tend = np.where(times <= tmax)[0][-1] + 1
    tslice = slice(tstart, tend, None)
    n_times = len(times[tslice])
    n_fft = n_times if n_fft is None else n_fft

    # Preparing frequencies of interest
    orig_freqs = np.fft.rfftfreq(n_fft, 1. / sfreq)
    freq_mask = (orig_freqs > fmin) & (orig_freqs < fmax)
    freqs = orig_freqs[freq_mask]
    n_freqs = len(freqs)

    if n_freqs == 0:
        raise ValueError('No discrete fourier transform results within '
                         'the given frequency window. Please widen either '
                         'the frequency window or the time window')

    window_fun, eigvals, n_tapers, mt_adaptive = mne.time_frequency.csd._compute_csd_params(
        n_times, sfreq, mode, mt_bandwidth, mt_low_bias, mt_adaptive)

    csds_mean = np.zeros((epochs.shape[1], epochs.shape[1], n_freqs),
                         dtype=complex)

    # Picking frequencies of interest
    freq_mask_mt = freq_mask[orig_freqs >= 0]

    # Compute CSD for each epoch
    n_epochs = 0
    if parallel is None:
        for epoch in epochs:
            epoch = epoch[:, tslice]
            # Calculating Fourier transform using multitaper module
            csds_epoch = mne.time_frequency.csd._csd_array(epoch, sfreq, window_fun, eigvals, freq_mask,
                                                           freq_mask_mt, n_fft, mode, mt_adaptive)

            # Scaling by sampling frequency for compatibility with Matlab
            csds_epoch /= sfreq
            csds_mean += csds_epoch
            n_epochs += 1

        csds_mean /= n_epochs
        csd_mean_fsum = np.sum(csds_mean, 2)

    else:
        print 'Doing CSD in parallel'
        csd_epochs = parallel(
            delayed(csd_one_epoch)(
                epoch, sfreq, window_fun, eigvals, freq_mask, freq_mask_mt, n_fft, mode, mt_adaptive)
            for epoch in epochs[:, :, tslice])
        N = len(csd_epochs)
        csd_mean_fsum = csd_epochs.pop()
        for ce in csd_epochs:
            csd_mean_fsum += ce
        csd_mean_fsum /= float(N)
        del csd_epochs

        # Summing over frequencies of interest or returning a list of separate CSD
        # matrices for each frequency
    csd = mne.time_frequency.csd.CrossSpectralDensity(csd_mean_fsum, ch_names, None,
                                                      None,
                                                      freqs=freqs, n_fft=n_fft)
    return csd


def csd_one_epoch(epoch, sfreq, window_fun, eigvals, freq_mask, freq_mask_mt, n_fft, mode, mt_adaptive):
    # Calculating Fourier transform using multitaper module
    csds_epoch = mne.time_frequency.csd._csd_array(epoch, sfreq, window_fun,
                                                   eigvals, freq_mask,
                                                   freq_mask_mt, n_fft, mode,
                                                   mt_adaptive)

    # Scaling by sampling frequency for compatibility with Matlab
    csds_epoch /= sfreq
    return csds_epoch
