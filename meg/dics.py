
import numpy as np
from mne.time_frequency import csd_epochs
from conf_analysis.meg import source_recon as sr

from copy import deepcopy

from scipy import linalg

from mne.beamformer._lcmv import (_prepare_beamformer_input, _setup_picks,
                                  _reg_pinv)
from mne.externals import six


def get_tfr(subject):
    from glob import glob
    from mne.time_frequency.tfr import read_tfrs
    from mne.time_frequency import EpochsTFR
    files = glob('/home/nwilming/conf_meg/S%i/SUB%i*stimulus*-tfr.h5' %
                 (subject, subject))
    etfr = [read_tfrs(files.pop())[0]]
    for fname in files:
        etfr.append(read_tfrs(fname)[0])    
    data = np.concatenate([e.data for e in etfr], 0)
    return EpochsTFR(etfr[0].info, data, etfr[0].times, etfr[0].freqs)


def dics(epochs, tfr, f, times, f_smooth, t_smooth, subject):
    '''
    epochs - TFR Epochs object    
    f - frequency in Hz
    times - array of time points
    f_smooth - Frequency smoothing of TFR estimate (+-f_smooth)
    t_smooth - Temporal smoothing of TFR estmeate (+-t_smooth)
    '''
    fmin, fmax = f - f_smooth, f + f_smooth

    forward, bem, source, trans = sr.get_leadfield(subject)

    noise_csd = csd_epochs(epochs, 'multitaper', fmin, fmax,
                           fsum=True, tmin=0.75 - 2 * t_smooth,
                           tmax=0.75)

    source_pow = np.nan * np.ones((forward['nsource'],
                                   tfr.data.shape[0],
                                   tfr.data.shape[3]))

    id_freq = np.argmin(np.abs(tfr.freqs - f))

    idx = ((times[0]+t_smooth) < times) & (times < (times[1]-t_smooth))
    for i, t in enumerate(times[idx]):
        # Construct filter
        data_csd = csd_epochs(epochs, 'multitaper', fmin, fmax,
                              fsum=True, tmin=t - t_smooth, tmax=t + t_smooth)
        dics_filter = make_dics(epochs.info, forward, noise_csd, data_csd)
        A = dics_filter['weights']

        for epoch in range(tfr.data.shape[0]):
            Xsensor = tfr.data[epoch, :, id_freq,  i]  # 269; Avoid copy
            Xsource = np.dot(A, Xsensor)  # 8k = (8k x 269) * (269,)
            source_pow[:, epoch, i] = Xsource * np.conj(Xsource)
    return forward, bem, source, trans, source_pow


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
                   src=deepcopy(forward['src']))

    return filters
