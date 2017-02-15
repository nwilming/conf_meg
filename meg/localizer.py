import numpy as np
import mne
from pymeg import preprocessing, tfr
from sklearn import cluster, neighbors
from sklearn.metrics import pairwise
import glob
from joblib import Memory
from conf_analysis.behavior import metadata
memory = Memory(cachedir=metadata.cachedir, verbose=0)

try:
    params = tfr.params_from_json('all_tfr150_parameters.json')
except IOError:
    params=None


@memory.cache
def get_localizer_epochs(filename, reject=dict(mag=4e-12)):
    raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
    sf = float(raw.info['sfreq'])
    mapping = {50:('con_change', 0), 64:('stim_onset', 0), 160:('start', 0), 161:('end',0)}
    meta, timing = preprocessing.get_meta(raw, mapping, {}, 160, 161)
    if len(meta)==0:
        return None
    tmin, tmax = (timing.min().min()/sf)-5, (max(timing.max().max())/sf)+5
    raw = raw.crop(tmin=max(0, tmin),
                   tmax=min(tmax, raw.times[-1]))
    raw.load_data()
    raw.notch_filter(np.arange(50, 251, 50), n_jobs=4)
    meta, timing = preprocessing.get_meta(raw, mapping, {}, 160, 161)

    l = len(timing)
    events = np.vstack([timing.stim_onset_time, [0]*l, [1]*l]).T
    e = mne.Epochs(raw, events=events.astype(int),
        tmin=-1.25, tmax=1.75, reject=reject,
        )
    del raw
    return e


def get_localizer(snum, reject=dict(mag=4e-12)):
    files = glob.glob('/home/nwilming/conf_meg/raw/s%02i-*.ds'%snum)
    files += glob.glob('/home/nwilming/conf_meg/raw/S%02i-*.ds'%snum)
    epochs = [get_localizer_epochs(f) for f in files]
    epochs = [e for e in epochs if e is not None]
    dt = epochs[0].info['dev_head_t']
    for e in epochs:
        e.info['dev_head_t'] = dt
        e.reject = reject
        e.load_data()
    return mne.concatenate_epochs([e for e in epochs if len(e)>0])


@memory.cache
def get_localizer_power(snum, reject=dict(mag=4e-12), params=params):
    print 'Getting localizer'
    epochs = get_localizer(snum, reject=reject)
    epochs.pick_channels([ch for ch in epochs.ch_names if ch.startswith('M')])
    epochs.resample(450)
    print 'Doing power calculation'
    power = get_power(epochs, params=params)
    return power

def get_power(epochs, params=params):
    return mne.time_frequency.tfr_multitaper(epochs,
        params['foi'], params['cycles'], return_itc=False, n_jobs=4)


def get_sensor_selection(power, freqs=(50, 110),
        period=(0.05, 0.3),  cutoffs=[2.5, 5, 7.5, 10]):

    # Discard first and last bits of data
    id_use = (-0.5 < power.times) & (power.times < 1.1)
    id_freqs = (freqs[0] < power.freqs) & (power.freqs < freqs[1])
    data = power.data[:, id_freqs, :][:, :, id_use]
    times = power.times[id_use]
    id_base = (-0.25 < times) & (times < 0)
    base_mean = data[:, :, id_base].mean(2)[:, :, np.newaxis]
    base_std = data[:, :, id_base].std(2)[:, :, np.newaxis]
    data = (data-base_mean) / base_std
    data = data.mean(1)
    sensor_id, sensor_names = [], []
    start, end = period
    id_t = (start<= times) & (times < end)
    for cutoff in cutoffs:
        sensor_id.append(np.where(abs(data[:, id_t].mean(1)) > cutoff)[0])
        sensor_names.append([power.ch_names[sid] for sid in sensor_id[-1]])
    return data, times, sensor_id, sensor_names


def cluster_sensors(power, n_clusters=2, freq=slice(30, 200)):
    '''
    Use hierarchical agglomerative clustering (ward) to select channels.
    '''
    freqs = power.freqs
    pos = np.asarray([ch['loc'][:3] for ch in power.info['chs']])
    averages = (power.copy()
                     .apply_baseline((-0.2, 0), mode='zscore')
                     .crop(tmin=0.00, tmax=1))

    id_freqs = (freq.start<freqs) & (freqs < freq.stop)
    data = averages.data[:, id_freqs, :]
    dims = data.shape
    dists = pairwise.pairwise_distances(data.reshape(dims[0], dims[1]*dims[2]), metric='l1')

    N = neighbors.kneighbors_graph(pos, n_neighbors=5)
    cl = cluster.AgglomerativeClustering(n_clusters=n_clusters, connectivity=N, linkage='ward')
    cl.fit(dists)
    labels = cl.labels_
    clusters = [np.where(cl.labels_==k)[0] for k in np.unique(cl.labels_)]
    freq_resp = [averages.data.mean(-1)[c,:].mean(0) for c in clusters]
    f_roi = [freqs[np.argmax(fr)] for fr in freq_resp]
    ch_names = [[power.ch_names[k] for k in clt] for clt in clusters]
    return clusters, ch_names, f_roi, freq_resp, dists
