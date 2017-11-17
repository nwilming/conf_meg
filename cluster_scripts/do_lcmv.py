import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import lcmv
import numpy as np


def lcmvfilename(subject, session):
    filename = '/home/nwilming/conf_meg/S%i-SESS%i-lcmv.hdf' % (
        subject, session)
    return filename


def execute(subjid, session, kws):
    if 'lowest_freq' not in kws.keys():
        kws['lowest_freq'] = None
    epochs, estcs, localizer, lstcs = lcmv.extract(subjid, session, **kws)
    filename = lcmvfilename(subjid, session)
    if epochs is not None:
        epochs.to_hdf(filename, 'epochs')
    #if localizer is not None:
    #    localizer.to_hdf(filename, 'localizer')
    print estcs
    lcmv.make_averaged_sr(estcs, subjid, session, kws['lowest_freq'], prefix='')
    #lcmv.make_averaged_sr(lstcs, subjid, session, kws['lowest_freq'], prefix='loc')


def list_tasks(**kws):
    for f in [1, 2]:  #, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        for session in [0,1,2,3]:
            yield f, session, kws


if __name__ == '__main__':
    for x in list_tasks():
        print x
