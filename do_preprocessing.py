import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import preprocessing
import cPickle
import mne, locale
locale.setlocale(locale.LC_ALL, "en_US")

import os
import datetime
def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def list_tasks(older_than='now'):
    if older_than is 'now':
        older_than = datetime.datetime.today()
    else:
        older_than = datetime.datetime.strptime(older_than, '%Y%m%d')

    block_map = cPickle.load(open('meg/blockmap.pickle'))
    for snum in range(1, 16):
        for session in range(0,4):
            map_blocks = dict((v,k) for k, v in block_map[snum][session].iteritems())
            for block in map_blocks.keys():
                if block not in map_blocks.keys():
                    print (snum, session, block), map_blocks
                else:
                    block_in_raw, block_in_experiment = map_blocks[block], block
                    filename = '/home/nwilming/conf_meg/S%i/SUB%i_S%i_B%i_stimulus-epo.fif.gz'%(snum, snum, session, block_in_experiment)
                    try:
                        mod_date = modification_date(filename)
                        if mod_date>older_than:
                            continue
                    except OSError:
                        pass
                    yield (snum, session, block_in_raw, block_in_experiment)


def execute(x):
    print 'Starting task:', x
    res = preprocessing.one_block(*x)
    print 'Ended task:', x
