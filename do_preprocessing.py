import sys
sys.path.append('/home/nwilming/')
from conf_analysis.meg import artifacts, preprocessing
from conf_analysis.behavior import empirical, metadata, keymap
from distributed import Executor, as_completed
from distributed import diagnostics
import cPickle
import mne, locale
locale.setlocale(locale.LC_ALL, "en_US")

executor = Executor("172.18.101.120:8786")


block_map = cPickle.load(open('meg/blockmap.pickle'))

results = []

q = []

def expand(x):
    print 'Starting task:', x
    res = preprocessing.one_block(*x)
    print 'Ended task:', x

for snum in range(1, 16):
    for session in range(0,4):
        #filename = metadata.get_raw_filename(snum, session)
        #raw = mne.io.read_raw_ctf(filename, system_clock='ignore')

        map_blocks = dict((v,k) for k, v in block_map[snum][session].iteritems())
        for block in map_blocks.keys():
            block_in_raw, block_in_experiment = map_blocks[block], block
            q.append((snum, session, block_in_raw, block_in_experiment))

results = executor.map(expand, q)
diagnostics.progress(results)
executor.gather(results)
