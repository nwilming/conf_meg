import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import lcmv
import numpy as np
import datetime
import os
from pymeg import tfr


def lcmvfilename(subject, session, F, tuning=None):
    if tuning is None:
        filename = '/home/nwilming/conf_meg/S%i-SESS%i-F%f-lcmv.hdf' % (
            subject, session, np.around(F, 2))
    else:
        filename = '/home/nwilming/conf_meg/S%i-SESS%i-F%f-tune%i-lcmv.hdf' % (
            subject, session, np.around(F, 2), tuning)
    return filename


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def execute(subjid, session, kws):
    if 'lowest_freq' not in kws.keys():
        kws['lowest_freq'] = None
    p = None
    F = None
    if 'F' in kws.keys() and kws['F'] == 'determine':
        cycles, tb = 8., 2.
        avg = lcmv.load_tfr(subjid, session)
        idx = lcmv.select_channels_by_gamma(avg)
        F_base = np.around(lcmv.freq_from_tfr(avg, idx), 0)
        tds = [-7.5, -5., -2.5, 0, 2.5, 5, 7.5]
        for i, Fdelta in enumerate(tds):
            Fact = F + Fdelta
            print('F determined to be', Fact)
            p = lcmv.get_power_estimator(Fact, cycles, tb, sf=600.)
            _, _, ts, fs = tfr.taper_data(
                Fact, cycles=cycles, time_bandwidth=tb)[0]
            lowest_freq = Fact - 1.5 * fs
            run(subjid, session, p, lowest_freq, Fact, tuning=i)
    elif 'F' in kws.keys():
        params = tfr.params_from_json(
            '/home/nwilming/conf_analysis/required/all_tfr150_parameters.json')
        F = float(kws['F'])
        foi, cycles, tb = params['foi'], params[
            'cycles'], params['time_bandwidth']
        fidx = np.argmin(np.abs(np.array(foi) - F))
        F = foi[fidx]
        c = cycles[fidx]
        print('Constructing TFR filter with F=',
              F, 'cycles', c, 'time_bandwidth', tb)
        p = lcmv.get_power_estimator(F, c, tb, sf=600.)
        del kws['F']
        run(subjid, session, p, kws['lowest_freq'], F)
    else:
        run(subjid, session, None, kws['lowest_freq'], F)


def run(subjid, session, p, lowest_freq, F, tuning=None):
    accum = lcmv.AccumSR(subjid, session,
                         lowest_freq, F, prefix='')
    epochs = lcmv.extract(
        subjid, session, func=p, accumulator=accum, lowest_freq=lowest_freq)
    filename = lcmvfilename(subjid, session, F, tuning=tuning)
    if epochs is not None:
        epochs.to_hdf(filename, 'epochs')
    accum.save_averaged_sr()
    return epochs


def list_tasks(**kws):
    for f in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        for session in [0, 1, 2, 3]:
            if 'older_than' in kws.keys():
                older_than = kws['older_than']
                del kws['older_than']
                filename = lcmvfilename(f, session)
                if len(older_than) == 8:
                    older_than = datetime.datetime.strptime(
                        older_than, '%Y%m%d')
                else:
                    older_than = datetime.datetime.strptime(
                        older_than, '%Y%m%d%H%M')
                try:
                    mod_date = modification_date(filename)
                    if mod_date > older_than:
                        continue
                except OSError:
                    pass
            yield f, session, kws


if __name__ == '__main__':
    for x in list_tasks():
        print x
