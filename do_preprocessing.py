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


#data = empirical.load_data()
#data = empirical.data_cleanup(data)

block_map = cPickle.load(open('meg/blockmap.pickle'))

results = []
for snum in range(1, 6):
    for session in range(4):
        #filename = metadata.get_raw_filename(snum, session)
        #raw = mne.io.read_raw_ctf(filename, system_clock='ignore')

        map_blocks = dict((v,k) for k, v in block_map[snum][session].iteritems())
        for block in range(0, 5):
            block_in_raw, block_in_experiment = map_blocks[block], block
            #results.append(executor.submit(sum, [1,2,3,4,]))

            results.append(executor.submit(preprocessing.one_block, snum, session, block_in_raw, block_in_experiment))


diagnostics.progress(results)

for result in as_completed(results):
    print result.result()
