'''
Determine mapping of blocks in data files to blocks in experiment.
'''
from conf_analysis.meg import artifacts, preprocessing, keymap
from conf_analysis.behavior import empirical, metadata
import mne, locale
import numpy as np
import cPickle
locale.setlocale(locale.LC_ALL, "en_US")

try:
    block_map = cPickle.load(open('blockmap.pickle'))
except IOErrror:
    block_map = {}

for snum in range(5, 6    ):
    filenames = [metadata.get_raw_filename(snum, b) for b in range(4)]
    block_map[snum] = {}
    for session, filename in enumerate(filenames):
        block_map[snum][session] = {}
        raw = mne.io.read_raw_ctf(filename, system_clock='ignore')

        trials = preprocessing.blocks(raw)
        trl, bl = trials['trial'], trials['block']
        bcnt = 1
        for b in np.unique(bl):
            if len(trl[bl==b]) >= 75:
                block_map[snum][session][b] = bcnt
                bcnt +=1

print block_map
cPickle.dump(block_map, open('blockmap.pickle', 'w'))
