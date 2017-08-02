from conf_analysis.meg import artifacts, preprocessing, keymap
from conf_analysis.behavior import empirical, metadata
import mne, locale
import numpy as np

locale.setlocale(locale.LC_ALL, "en_US")

filename = '/Volumes/dump/conf_data/raw/s02-01_Confidence_20151208_02.ds'

data = empirical.load_data()
data = empirical.data_cleanup(data)
data = data.query('snum==2 & day==20151208')


# Compute a unique hash for all entries in data
data.loc[:, 'hash'] =  [keymap.hash(x) for x in
                            data.loc[:, ('day', 'snum', 'block_num', 'trial')].values]

assert len(np.unique(data.loc[:, 'hash'])) == len(data.loc[:, 'hash'])

raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
#trials = preprocessing.blocks(raw)
#r, r_id = preprocessing.load_block(raw, trials, block)
raw.crop(tmin=813., tmax=913.)

#r, ants, artdefs = preprocessing.preprocess_block(r)

ab = artifacts.annotate_blinks(raw)
am, zm = artifacts.annotate_muscle(raw)
ac, zc = artifacts.annotate_cars(raw)
ar, zj = artifacts.annotate_jumps(raw)

meta, timing = preprocessing.get_meta(data, r, snum, block_cnt)


# Define stimulus epochs.
m, s = preprocessing.get_epoch(r, meta, timing,
                               event='stim_onset_t', epoch_time=(-.2, 1.5),
                               base_event='stim_onset_t', base_time=(-.2, 0))


from sklearn import svm
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

estimators = [('Scale', StandardScaler()), ('svm', svm.LinearSVC())]

clf = Pipeline(estimators)

y = m.response.values
scores = []
for t in range(e._data.shape[2]):
    X = e._data[:, :, -100]
    scores += [mean(cross_validation.cross_val_score(clf, X, y, cv=5))]
