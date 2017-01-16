import mne, locale
import numpy as np
import json
import cPickle

locale.setlocale(locale.LC_ALL, "en_US")

def describe_taper(foi=None, cycles=None, time_bandwidth=None, **kwargs):
    from tabulate import tabulate
    if len(np.atleast_1d(cycles))==1:
        cycles = [cycles]*len(foi)
    foi = np.atleast_1d(foi)
    cycles = np.atleast_1d(cycles)
    time = cycles/foi
    f_smooth = time_bandwidth/time
    data = zip(list(foi), list(cycles), list(time), list(f_smooth))
    print tabulate(data,   headers=['Freq', 'Cycles', 't. window', 'F. smooth'])


def params_from_json(filename):
    params = json.load(open(filename))
    assert('foi' in params.keys())
    assert('cycles' in params.keys())
    assert('time_bandwidth' in params.keys())
    return params


def tfr(filename, outstr='tfr.pickle', foi=None, cycles=None, time_bandwidth=None, decim=10, n_jobs=12, **kwargs):
    outname = filename.replace('-epo.fif.gz', outstr)
    epochs = mne.read_epochs(filename)
    power = mne.time_frequency.tfr_multitaper(epochs, foi, cycles,
        decim=decim, time_bandwidth=time_bandwidth, average=False, return_itc=False,
        n_jobs=12)
    print filename, '-->', outname
    cPickle.dump({'power': power,
                  'foi': foi,
                  'cycles': cycles,
                  'time_bandwidth': time_bandwidth,
                  'decim':decim,
                  'events':epochs.events}, open(outname, 'w'), 2)
    return True


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("epoch", help='Which epoch file?')
parser.add_argument("tfrdef", help="""TFR definition file. A json file that
contains parameters for a TFR analysis:
    - foi: list, frequencies of interest
    - cycles: list, number of cycles for each Frequency
    - time_bandwidth: int, number of tapers per Frequency
    - decim: int, decimation factor""",
                    type=str, default=1);
parser.add_argument("--describe", help='Describe TFR structure', dest='describe', action='store_true')
parser.add_argument("--dry", help='If true do not compute TFR', dest='dry', action='store_true')

parser.add_argument("--outstr", help='-epo.fif.gz in the epoch filename will be replaced by this.', type=str, default='-tfr.pickle')

args = parser.parse_args()

params = params_from_json(args.tfrdef)
if args.describe:
    describe_taper(**params)

print args.dry
if not args.dry:
    tfr(args.epoch, args.outstr, **params)
