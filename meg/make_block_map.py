'''
Determine mapping of blocks in data files to blocks in experiment.
'''
from conf_analysis.meg import artifacts, preprocessing, keymap
from conf_analysis.behavior import empirical, metadata
import mne, locale
import numpy as np
import cPickle
locale.setlocale(locale.LC_ALL, "en_US")

block_map = {}
for s in ['S%02i' in range(15)]:
    filenames = metadata.data_files[s]
    block_map[s] = {}
    for session in enumerate(filenames):
        block_map[s][session] = {}
        trials = blocks(raw)
        trl, bl = trials['trial'], trials['bl']
        bcnt = 0
        for b in np.unique(bl):
            if len(trl[bl==b]) == 500:
                block_map[s][session][b] = bcnt
                bcnt +1

print block_map
cPickle.dump(block_map, open('blockmap.pickle', 'w'))
