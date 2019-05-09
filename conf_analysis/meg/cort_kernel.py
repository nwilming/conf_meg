import logging
import os
import numpy as np
import pandas as pd

from os.path import join
from glob import glob

from pymeg import aggregate_sr as asr
from conf_analysis.behavior import metadata
from conf_analysis.behavior import empirical, kernels
from conf_analysis.meg import regress
from conf_analysis.meg import decoding_plots as dp, decoding_analysis as da

from joblib import Memory

if "TMPDIR" in os.environ.keys():
    memory = Memory(cachedir=os.environ["PYMEG_CACHE_DIR"])
    inpath = "/nfs/nwilming/MEG/sr_labeled/aggs"
    outpath = "/nfs/nwilming/MEG/sr_decoding/"
elif "RRZ_LOCAL_TMPDIR" in os.environ.keys():
    tmpdir = os.environ["RRZ_LOCAL_TMPDIR"]
    outpath = "/work/faty014/MEG/sr_labeled/aggs/"
    outpath = "/work/faty014/MEG/sr_decoding/"
    memory = Memory(cachedir=tmpdir)
else:
    inpath = "/home/nwilming/conf_meg/sr_labeled/aggs"
    outpath = "/home/nwilming/conf_meg/sr_decoding"
    memory = Memory(cachedir=metadata.cachedir)


def to_fluct(Xs, side):
    ids = side == 1
    mean_larger = Xs[ids, :].mean(0)[np.newaxis, :]
    mean_smalle = Xs[~ids, :].mean(0)[np.newaxis, :]
    Xs[ids, :] -= mean_larger
    Xs[~ids, :] -= mean_larger
    return Xs


def get_kernel(subject):
    K, C = get_kernels()
    return K.loc[subject, :], C.loc[subject, :]


@memory.cache()
def get_kernels():
    d = empirical.load_data()
    K = (
        d.groupby(["snum", "side"])
        .apply(kernels.get_decision_kernel)
        .groupby("snum")
        .mean()
    )  # dp.extract_kernels(dz)
    C = (
        d.groupby("snum")
        .apply(kernels.get_confidence_kernel)
        .stack()
        .groupby("snum")
        .mean()
    )  # dp.extract_kernels(dz)
    return K, C


def _par_get_cortical_kernel(*args, **kwargs):
    return get_cortical_kernel(*args, **kwargs)


@memory.cache()
def get_cortical_kernel(
    subject,
    hemi,
    cluster,
    latency,
    remove_contrast_induced_flucts=False,
    freq_band=[45, 65],
    ogl=True,
):

    X, tpc, freq, meta = build_design_matrix(
        subject, cluster, latency, hemi, freq_bands=freq_band, ogl=ogl
    )
    if remove_contrast_induced_flucts:
        contrast = np.stack(meta.contrast_probe)
        X = remove_contrast_induced_fluctuations(data, contrast)
    idside = meta.loc[:, "side"] == 1
    return (kernels.kernel(X[idside], meta.loc[idside, "response"])[0] - 0.5) - (
        kernels.kernel(X[~idside], meta.loc[~idside, "response"])[0] - 0.5
    )


def remove_contrast_induced_fluctuations(data, contrast):
    """
    Predict contrast induced fluctuations by means of linear regression.

    Data is n_trials x 10 matrix of power values
    contrast is n_trials x 10 matrix of contrast values

    """
    data = data.copy()
    for i in range(10):
        slope, intercept, _, _, _ = linregress(contrast[:, i], data[:, i])
        data[:, i] -= slope * data[:, i] + intercept
    return data


def correlate(
    subject, hemi, cluster, latency, peakslope=None, freq_band=[45, 65], ogl=False
):
    """
    Compute three different correlations with choice kernel:

    1) Contrast induced fluctuations
    This measures the profile of contrast decodability and correlates
    it with choice kernels.
    2) Internally induced fluctuations.
    This measures fluctuations of reconstructed power values after the
    effect of contrast is subtracted out.
    3) A mixture of both.
    This uses the overall power fluctuations and correlates them with
    the choice kernel.
    """
    k, c = get_kernel(subject)
    res = {"subject": subject, "cluster": cluster, "latency": latency}
    try:
        ck_int = get_cortical_kernel(
            subject,
            hemi,
            cluster,
            latency,
            freq_band=freq_band,
            ogl=ogl,
            remove_contrast_induced_fluctuations=True,
        )
        ck_all = get_cortical_kernel(
            subject,
            hemi,
            cluster,
            latency,
            freq_band=freq_band,
            ogl=ogl,
            remove_contrast_induced_fluctuations=False,
        )
        res["choice_corr_cif_removed"] = np.corrcoef(ck_int.ravel(), k.ravel())[0, 1]
        res["choice_corr_with_cif"] = np.corrcoef(ck_all.ravel(), k.ravel())[0, 1]

    except RuntimeError:
        # Area missing
        pass

    if peakslope is not None:
        # vfcpeak = dp.extract_peak_slope(ssd)
        pvt = peak.loc[:, cluster]
        assert pvt.shape == (10, 15)
        res["choice_corr_contrast_induced"] = np.corrcoef(
            K.loc[sub, :].values, pvt.loc[:, sub].values
        )[0, 1]
    return res


@memory.cache()
def build_design_matrix(
    subject, cluster, latency, hemi="Averaged", zscore=True, freq_bands=None, ogl=False
):
    if not ogl:
        filenames = glob(join(inpath, "S%i_*_%s_agg.hdf" % (subject, "stimulus")))
        data = asr.delayed_agg(filenames, hemi=hemi, cluster=cluster)()
    else:
        filenames = glob(
            join(inpath, "ogl/slimdown/ogl_S%i_*_%s_agg.hdf" % (subject, "stimulus"))
        )
        data = asr.delayed_agg(
            filenames, hemi="Pair", cluster=["%s_LH" % cluster, "%s_RH" % cluster]
        )().copy()

        data = data.groupby(["hemi", "trial", "freq"]).mean()
        data.head()
        hemi = np.asarray(data.index.get_level_values("hemi")).astype("str")
        trial = np.asarray(data.index.get_level_values("trial")).astype(int)
        freq = np.asarray(data.index.get_level_values("freq")).astype(int)
        # cl = hemi.copy()
        cl = [cluster] * len(hemi)
        index = pd.MultiIndex.from_arrays(
            [hemi, cl, trial, freq], names=["hemi", "cluster", "trial", "freq"]
        )
        data.index = index
        data.head()

    trial_index = data.index.get_level_values("trial")
    X, time_per_col = regress.prep_low_level_data(
        cluster, data, 0, latency, trial_index, freq_bands=freq_bands
    )
    cols = X.columns.values
    index = X.index.values
    X = X.values
    meta = da.preprocessing.get_meta_for_subject(subject, "stimulus")
    meta.set_index("hash", inplace=True)
    meta = meta.loc[trial_index]
    if zscore:
        X = (X - X.mean(0)) / X.std(0)

    return X, time_per_col, cols, meta
