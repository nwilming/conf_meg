import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import preprocessing, decoding
from conf_analysis.behavior import empirical, metadata, keymap

import cPickle
import mne, locale
import pandas as pd
import numpy as np


block_map = cPickle.load(open('meg/blockmap.pickle'))


def modification_date(filename):
    try:
        t = os.path.getmtime(filename)
        return datetime.datetime.fromtimestamp(t)
    except OSError:
        return datetime.datetime.strptime('19700101', '%Y%m%d')


def contrast_transform(row, j):
    row = np.vstack(np.asarray(row))
    if row.shape==(10,1):
        return row[j].mean()>0.5
    else:
        vals = np.vstack(row)
        return vals[:, j].mean(1)>0.5
def execute(x):
    print 'Starting task:', x
    sensors = decoding.sensors[x[3]]
    print sensors
    results = []
    for j, s in enumerate([slice(i, i+1) for i in range(10)] + [slice(0, 10)]):
        func = lambda x,y: (decoding.generalization_matrix(x, y, 12,
            slices='reshape'))
        res = decoding.apply_decoder(func, x[0], x[1], x[2],
            channels=sensors, label_func=lambda x: contrast_transform(x, j))
        res.loc[:, 'sensors'] = x[3]
        res.loc[:, 'snum'] = x[0]
        res.loc[:, 'epoch'] = x[1]
        res.loc[:, 'label'] = x[2]
        res.loc[:, 'slice'] = j
        results.append(res)
    results = pd.concat(results)

    results.to_hdf(x[4], 'decoding')
    #results = results.groupby([u'predict_time', u'train_time', 'sensors', u'snum', u'epoch', 'label']).mean().reset_index()
    return results


def list_tasks(older_than='now'):
    for snum in range(1, 16):
        for epoch in ['stimulus']:
            for label in ['contrast_probe']:
                for sensors in decoding.sensors.keys():
                    filename = 'results/decoding' + str(snum) + epoch + label + sensors + '.hdf'
                    yield (snum, epoch, label, sensors, filename)
