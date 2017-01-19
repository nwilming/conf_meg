import mne, locale
import numpy as np
import json
import cPickle
import os
import datetime

from pymeg import tfr

print tfr


def modification_date(filename):
    try:
        t = os.path.getmtime(filename)
        return datetime.datetime.fromtimestamp(t)
    except OSError:
        return datetime.datetime.strptime('19700101', '%Y%m%d')

locale.setlocale(locale.LC_ALL, "en_US")

outstr = 'tfr.hdf5'
params = tfr.params_from_json('all_tfr150_parameters.json')
tfr.describe_taper(**params)


def list_tasks(older_than='now'):
    import glob
    filenames = glob.glob('/home/nwilming/conf_meg/*/*stimulus-epo.fif.gz')
    filenames += glob.glob('/home/nwilming/conf_meg/*/*response-epo.fif.gz')
    filenames == glob.glob('/home/nwilming/conf_meg/*/*feedback-epo.fif.gz')

    if older_than == 'now':
        older_than = datetime.datetime.today()
    else:
        if len(older_than) == 8:
            older_than = datetime.datetime.strptime(older_than, '%Y%m%d')
        else:
            older_than = datetime.datetime.strptime(older_than, '%Y%m%d%H%M')

    for filename in filenames:

        mod_date = modification_date(filename)
        outname = filename.replace('-epo.fif.gz', outstr)
        try:
            mod_date = modification_date(filename)
            mod_out = modification_date(outname)
            if (mod_date>older_than) and (mod_out>older_than):
                continue
        except OSError:
            pass
        yield filename


def execute(filename):
    print 'Starting TFR for ', filename
    print params
    tfr.tfr(filename, outstr, **params)
    print 'Done with TFR for ', filename
