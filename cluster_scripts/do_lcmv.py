import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import lcmv
import numpy as np
import datetime
import os
from pymeg import tfr


def lcmvfilename(subject, session, BEM):
    filename = '/home/nwilming/conf_meg/sr_freq_labeled_%s/S%i-SESS%i-lcmv.hdf' % (
        BEM, subject, session)
    return filename


def srfilename(subject, session, BEM):
    filename = '/home/nwilming/conf_meg/source_recon_%s/S%i-SESS%i.stc' % (
        BEM, subject, session)
    return filename


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def execute(subjid, session, kws):
    p = None
    F = None
    BEM = 'three_layer'
    if 'BEM' in kws.keys():
        BEM = kws['BEM']

    estimators = (lcmv.get_broadband_estimator(),
                  lcmv.get_highF_estimator(),
                  lcmv.get_lowF_estimator())

    run(subjid, session, BEM=BEM)


def run(subjid, session, estimators, BEM='three_layer'):

    accum = lcmv.AccumSR(subjid,
                         srfilename(subjis, session, BEM),
                         'F', 55)

    filename = lcmvfilename(subjid, session, F, BEM)
    epochs = lcmv.extract(
        subjid, session, func=estimators, accumulator=accum,
        BEM=BEM)

    if epochs is not None:
        epochs.to_hdf(filename, 'epochs')
    accum.save_averaged_sr()
    return epochs


def list_tasks(**kws):
    for BEM in ['three_layer']:
        if BEM == 'three_layer':
            subs = [3]
        else:
            subs = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        for f in subs:
            for session in [0, 1, 2, 3]:
                if 'F' in kws.keys() and kws['F'] == 'determine':
                    yield f, session, kws
                else:
                    for F in range(5, 75, 5):
                        params = kws.copy()
                        params['F'] = F
                        params['BEM'] = BEM
                        yield f, session, params

if __name__ == '__main__':
    for x in list_tasks():
        print x
