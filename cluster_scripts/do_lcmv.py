import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import lcmv
import numpy as np


def lcmvfilename(subject):
    filename = '/home/nwilming/conf_meg/S%i-lcmv.hdf' % (
        subject)
    return filename


def execute(x):
    subjid, filter = x
    epochs, localizer = lcmv.extract(subjid, localizer_only=filter)
    filename = lcmvfilename(subjid)
    if epochs is not None:
        epochs.to_hdf(filename + 'epochs', 'epochs')
        epochs.to_hdf(filename, 'epochs')
    localizer.to_hdf(filename, 'localizer')
    localizer.to_hdf(filename + 'localizer', 'localizer')


def list_tasks(older_than='now', filter=None):
    for f in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        yield f, filter


if __name__ == '__main__':
    for x in list_tasks():
        print x
