'''
Analyze behavior of single trials
'''
import os
from os.path import join
from glob import glob
from conf_analysis.behavior import metadata
from conf_analysis.meg import preprocessing
from pymeg import aggregate_sr as asr

from joblib import Memory


if 'TMPDIR' in os.environ.keys():
    memory = Memory(cachedir=os.environ['PYMEG_CACHE_DIR'])
    inpath = '/nfs/nwilming/MEG/sr_labeled/aggs'
    outpath = '/nfs/nwilming/MEG/sr_decoding/'
elif 'RRZ_LOCAL_TMPDIR' in os.environ.keys():
    tmpdir = os.environ['RRZ_LOCAL_TMPDIR']
    outpath = '/work/faty014/MEG/sr_labeled/aggs/'
    outpath = '/work/faty014/MEG/sr_decoding/'
    memory = Memory(cachedir=tmpdir)
else:
    inpath = '/home/nwilming/conf_meg/sr_labeled/aggs'
    outpath = '/home/nwilming/conf_meg/sr_decoding'
    memory = Memory(cachedir=metadata.cachedir)


def get_trials(subject, epoch, cluster, hemi):
    filenames = glob(join(inpath, 'S%i_*_%s_agg.hdf' % (subject, epoch)))
    meta = preprocessing.get_meta_for_subject(subject, 'stimulus')
    agg = asr.delayed_agg(filenames, hemi=hemi, cluster=cluster)()
    return meta, agg


def trials_by_column(data, col):
    '''
    Takes data, df that contains all trials for one cluster, 
    and sorts by series col.
    '''
    # Make sure data has only trials as index
    names = data.index.names
    for name in names:
        if name == 'trial':
            continue
        data.index = data.index.droplevel(name)

    svals = col.index
    data = data.loc[svals, :]
    assert(all(data.index.values == svals.values))
    return data


def trials_by_index(data, col):
    # Make sure data has only trials as index
    names = data.index.names
    for name in names:
        if name == 'trial':
            continue
        data.index = data.index.droplevel(name)

    svals = col.index
    data = data.loc[svals, :]
    assert(all(data.index.values == svals.values))
    return data


def kernel(data, choices):
    '''
    Compute ROC AUC for choice weights.
    '''
    from sklearn.metrics import roc_curve_auc
    kernel = [roc_curve_auc(choices, column) for column in data]
    return kernel
