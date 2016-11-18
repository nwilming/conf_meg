import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import artifacts, preprocessing
from conf_analysis.behavior import empirical, metadata, keymap
import glob
import mne, locale
import pandas as pd
import numpy as np
from pylab import *
import cPickle

from mne.time_frequency import tfr
locale.setlocale(locale.LC_ALL, "en_US")

es = glob.glob('/home/nwilming/conf_meg/S4/*stimulus*.gz')


def describe_taper(freqs, n_cycles, time_bandwidth, sf):
    if len(atleast_1d(n_cycles))==1:
        n_cycles = [n_cycles]*len(freqs)
    for f, c in zip(freqs, n_cycles):
        time = c/float(f)
        f_smooth = time_bandwidth/time
        print 'Frequency: %iHz'%f, 'Filterlength: %2.1fs'%time, 'Frequency moothing: %2.1fHz'%f_smooth

foi = arange(50, 105, 5)
cycles = 0.1*foi

for filename in es:
    outname = filename.replace('-epo.fif.gz', '-gamma_tfr.hdr')

    epochs = mne.read_epochs(filename)
    power = mne.time_frequency.tfr_multitaper(epochs, foi, cycles,
        decim=5, time_bandwidth=2, average=False, return_itc=False, n_jobs=12)
    print filename, '-->', outname
    cPickle.dump(power, open(outname, 'w'), 2)
