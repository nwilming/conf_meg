import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import preprocessing, decoding
from conf_analysis.behavior import empirical, metadata, keymap

import cPickle
import mne, locale
import pandas as pd
import numpy as np
import datetime
import os
block_map = cPickle.load(open('meg/blockmap.pickle'))


def modification_date(filename):
    try:
        t = os.path.getmtime(filename)
        return datetime.datetime.fromtimestamp(t)
    except OSError:
        return datetime.datetime.strptime('19700101', '%Y%m%d')

epoch_time = {'stimulus':(-0.5, 1.25)}

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
    results = []
    for j, s in enumerate([slice(i, i+1) for i in range(10)] + [slice(0, 10)]):
        func = lambda x,y: (decoding.tfr_generalization_matrix(x, y, dt=1))
        res = decoding.tfr_apply_decoder(func, x[0], x[1], x[2],
            freq=(0, 100), time=epoch_time[x[1]], baseline=(-0.5, 0),
            channels=sensors, label_func=lambda x: contrast_transform(x, s))
        res.loc[:, 'sensors'] = x[3]
        res.loc[:, 'snum'] = x[0]
        res.loc[:, 'epoch'] = x[1]
        res.loc[:, 'label'] = x[2]
        res.loc[:, 'slice'] = j
        results.append(res)
    results = pd.concat(results)

    results.to_hdf(x[4], 'decoding')
    return results


def list_tasks(older_than='now'):
    if older_than == 'now':
        older_than = datetime.datetime.today()
    else:
        if len(older_than) == 8:
            older_than = datetime.datetime.strptime(older_than, '%Y%m%d')
        else:
            older_than = datetime.datetime.strptime(older_than, '%Y%m%d%H%M')

    for snum in range(1, 16):
        for epoch in ['stimulus']:
            for label in ['contrast_probe', ]:
                for sensors in decoding.sensors.keys():
                    filename = 'results/decoding' + str(snum) + epoch + label + sensors + 'tfr.hdf'
                    mod_date = modification_date(filename)
                    if mod_date>older_than:
                        continue
                    yield (snum, epoch, label, sensors, filename)
