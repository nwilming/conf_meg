import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import artifacts, preprocessing, decoding
from conf_analysis.behavior import empirical, metadata, keymap
from distributed import Executor, as_completed
from distributed import diagnostics
import cPickle
import mne, locale
import pandas as pd
import numpy as np
locale.setlocale(locale.LC_ALL, "en_US")

executor = Executor("172.18.101.120:8786")


block_map = cPickle.load(open('meg/blockmap.pickle'))

results = []

q = []

def expand(x):
    print 'Starting task:', x
    sensors = decoding.sensors[x[3]]
    print sensors
    func = lambda x,y: decoding.generalization_matrix(x, y, 12, slices=np.mean)
    res = decoding.apply_decoder(func, x[0], x[1], x[2], channels=sensors)
    res.loc[:, 'sensors'] = x[3]
    res.loc[:, 'snum'] = x[0]
    res.loc[:, 'epoch'] = x[1]
    res.loc[:, 'label'] = x[2]
    filename = 'results/decoding' + str(x[0]) + x[1] + x[2] + x[3] + '.hdf'
    res.to_hdf(filename, 'decoding')
    res = res.groupby([u'predict_time', u'train_time', 'sensors', u'snum', u'epoch', 'label']).mean().reset_index()
    return res


for snum in range(1, 16):
    for epoch in ['stimulus', 'response', 'feedback']:
        for label in ['side', 'correct', 'noise_sigma', 'conf_r1', 'conf_rm1']:
            for sensors in decoding.sensors.keys():
                q.append((snum, epoch, label, sensors))

results = executor.map(expand, q)
diagnostics.progress(results)
results = pd.concat(executor.gather(results))
results = results.groupby([u'predict_time', u'train_time', u'snum', u'epoch', 'label']).mean().reset_index()
results.to_hdf('generalization_across_time_additional.hdf', 'gat')
