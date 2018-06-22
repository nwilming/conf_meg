from pymeg import parallel


import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import preprocessing
import pickle
import mne
import locale
locale.setlocale(locale.LC_ALL, "en_US")

import os
import datetime


def list_tasks(**kws):

    block_map = pickle.load(
        open('/home/nwilming/conf_analysis/required/blockmap.pickle'))
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
                    yield (snum, session, block_in_raw, block_in_experiment)

for parameters in list_tasks():
    parallel.pmap(preprocessing.one_block, [parameters], walltime=5,
                  memory=20, nodes='1:ppn=2', name='RESP_PREP')
