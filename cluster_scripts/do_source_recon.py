import sys
sys.path.append('/home/nwilming/')
import glob
from pymeg import tfr
from conf_analysis.meg import source_recon as sr, preprocessing, dics
import zlib
import cPickle as pickle
from joblib import dump, load
import numpy as np
import json

def get_subject(subjid, F):
    '''
    Fetches all available data for a subject.
    '''
    F, _, _ = get_smoothing(F)

    filename = '/home/nwilming/conf_meg/S%i-%03.2iHz-power.memmap' % (
        subjid, F)
    meta = dics.get_metas_for_tfr(subjid)
    sp_shape = dics.output_shape(subjid, meta, (-0.75, 1.4), (F, F))
    source_pow = np.memmap(filename, dtype='float32', mode='r',
                           shape=sp_shape)
    forward, bem, source, trans = sr.get_leadfield(subjid)
    verts = [source[0]['vertno'], source[1]['vertno']]
    # Also load sidecar
    sidecar = json.load(open(filename.replace('memmap', 'sidecar.json')))
    times = sidecar['times']
    index = sidecar['index']
    index = dict((int(e), i) for e, i in index.iteritems())
    return F, meta, source_pow, forward, verts, times, index


def csdfilename(subject, hz, compression='.z'):
    filename = '/home/nwilming/conf_meg/S%i-%03.2iHz-dics-filters.pickle%s' % (
        subject, hz, compression)
    return filename


def get_smoothing(F):
    params = tfr.params_from_json(
        '/home/nwilming/conf_analysis/required/all_tfr150_parameters.json')
    return tfr.get_smoothing(F, **params)


def make_csd(subjid, F):
    print('Starting task:', subjid)
    epochs, meta = preprocessing.get_epochs_for_subject(subjid, 'stimulus')
    epochs.times = epochs.times - 0.75
    epochs = epochs.apply_baseline((-0.25, 0))

    tfrepochs = dics.get_tfr(subjid, n_blocks=1)
    F, t_smooth, f_smooth = get_smoothing(F)

    f, noise, data = dics.make_csds(epochs, F,
                                    [0, 1], [-0.5, 0], f_smooth, subjid)

    filename = csdfilename(subjid, F)
    dump({f: (noise, data)}, filename)


def apply_filter(subjid, F):
    F, t_smooth, f_smooth = get_smoothing(F)
    #tfrepochs = dics.get_tfr(subjid)
    meta = dics.get_metas_for_tfr(subjid)
    filtername = csdfilename(subjid, F)
    filters = load(filtername)
    noise_csd, data_csd = filters[F]

    filename = '/home/nwilming/conf_meg/S%i-%03.2iHz-power.memmap' % (
        subjid, F)

    sp, epoch_order = dics.apply_dics_filter(
        data_csd, F, meta, filename, subjid, n_jobs=12)
    # epoch order is a list of (index, epoch) tuples
    
    index = np.concatenate([i for i, e, t in epoch_order])
    epochs = np.concatenate([e for i, e, t in epoch_order])

    times = epoch_order[0][-1]
    epoch_order = dict((e, i) for e, i in zip(epochs, index))
    epoch_order = {'index':epoch_order, 'times':times}
    import json
    sidecar = filename.replace('memmap', 'sidecar.json')
    json.dump(epoch_order, open(sidecar, 'w'))


def convert_to_df(sujid, F):
    F, t_smooth, f_smooth = get_smoothing(F)
    filename = '/home/nwilming/conf_meg/S%i-%03.2iHz-power.memmap' % (
        subjid, F)
    dics.power_to_label_dataframe(subjid, filename)


def execute(x):
    subjid, F, task = x
    if (task is None) or (task == 'make'):
        make_csd(subjid, F)
    if (task is None) or (task == 'apply'):
        apply_filter(subjid, F)


def list_tasks(older_than='now', filter=None):
    freqs = [15.96076665,   17.04796317,   18.20921605,
             19.44956977,   20.77441243,   22.18949915,
             23.70097707,   25.31541203,   27.03981716,
             28.88168327,   30.84901142,   32.95034769,
             35.19482029,   37.5921792,    40.15283856,
             42.88792186,   45.80931031,   48.92969443,
             52.26262916,   55.82259278,   59.62504977,
             63.686518,     68.02464049]
    for f in [4, ]:
        for freq in freqs[3::3]:
            print f, freq, filter
            yield f, freq,  filter


if __name__ == '__main__':
    for x in list_tasks():
        print x
