import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import lcmv
import numpy as np
import datetime
import os
from pymeg import tfr


def lcmvfilename(subject, session, F, BEM):
    filename = '/home/nwilming/conf_meg/sr_freq_labeled_%s/S%i-SESS%i-F%f-lcmv.hdf' % (
        BEM, subject, session, np.around(F, 2))
    return filename


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def execute(subjid, session, kws):
    if 'lowest_freq' not in kws.keys():
        kws['lowest_freq'] = None
    force = False
    if 'force' in kws.keys():
        force = True
    p = None
    F = None
    BEM = 'three_layer'
    if 'BEM' in kws.keys():
        BEM = kws['BEM']
    if 'F' in kws.keys():
        cycles, tb = 8., 2.
        Fact = float(kws['F'])
        print('Constructing TFR filter with F=',
              Fact, 'cycles', cycles, 'time_bandwidth', tb)
        p = lcmv.get_power_estimator(Fact, cycles, tb, sf=600.)
        _, _, ts, fs = tfr.taper_data(
            Fact, cycles=cycles, time_bandwidth=tb)[0]
        lowest_freq = None # No high pass filtering before estimation. Old: Fact - 1.5 * fs 
        run(subjid, session, p, lowest_freq, Fact, BEM=BEM)
    else:
        run(subjid, session, None, kws['lowest_freq'], F, BEM=BEM)


def run(subjid, session, p, lowest_freq, F, BEM='three_layer'):

    accum = lcmv.AccumSR(subjid, session,
                         lowest_freq, F, BEM=BEM, prefix=BEM)
    filename = lcmvfilename(subjid, session, F, BEM)
    epochs = lcmv.extract(
        subjid, session, func=p, accumulator=accum,
        lowest_freq=lowest_freq, BEM=BEM)
    # else:
    #    return
    if epochs is not None:
        epochs.to_hdf(filename, 'epochs')
    accum.save_averaged_sr()
    return epochs


def list_tasks(**kws):
    for BEM in ['single_layer']:
        if BEM == 'three_layer':
            subs = [6, 7]
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
