import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import preprocessing
import pickle
import mne
import locale
locale.setlocale(locale.LC_ALL, "en_US")

import os
import datetime


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def list_tasks(**kws):

    if 'older_than' not in kws.keys() or kws['older_than'] == 'now':        
        older_than = datetime.datetime.today()
    else:
        if len(older_than) == 8:
            older_than = datetime.datetime.strptime(older_than, '%Y%m%d')
        else:
            older_than = datetime.datetime.strptime(older_than, '%Y%m%d%H%M')

    block_map = pickle.load(
        open('/home/nwilming/conf_analysis/meg/blockmap.pickle'))
    for snum in range(1, 16):            
        for session in range(0, 4):
            map_blocks = dict((v, k)
                              for k, v in block_map[snum][session].items())
            for block in list(map_blocks.keys()):
                if block not in list(map_blocks.keys()):
                    print((snum, session, block), map_blocks)
                else:
                    block_in_raw, block_in_experiment = map_blocks[
                        block], block
                    filenames = ['/home/nwilming/conf_meg/S%i/SUB%i_S%i_B%i_stimulus-epo.fif.gz' % (snum, snum, session, block_in_experiment),
                                 '/home/nwilming/conf_meg/S%i/SUB%i_S%i_B%i_response-epo.fif.gz' % (
                                     snum, snum, session, block_in_experiment),
                                 '/home/nwilming/conf_meg/S%i/SUB%i_S%i_B%i_feedback-epo.fif.gz' % (snum, snum, session, block_in_experiment)]
                    if 'filter' in kws:
                        if not any([kws['filter'] in f for f in filenames]):
                            #print kws['filter'], filenames
                            continue
                    try:
                        mod_dates = [modification_date(
                            filename) for filename in filenames]

                        if all([mod_date > older_than for mod_date in mod_dates]):
                            continue

                    except OSError as e:
                        print(e)
                        pass
                    yield (snum, session, block_in_raw, block_in_experiment)


def execute(*x):
    print('Starting task:', x)
    res = preprocessing.one_block(*x)
    print('Ended task:', x)
