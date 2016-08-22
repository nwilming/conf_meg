from conf_analysis.meg import artifacts, preprocessing, keymap
from conf_analysis.behavior import empirical, metadata
from distributed import Executor, as_completed
import cPickle

executor = Executor()


data = empirical.load_data()
data = empirical.data_cleanup(data)

block_map = cPickle.load(open('meg/blockmap.pickle'))

results = []
for snum in range(1, 6):
    for session in range(4):
        map_blocks = dict((v,k) for k, v in block_map[snum][session].iteritems())
        for block in range(1, 6):
            block_in_raw, block_in_experiment = map_blocks[block], block

            results.append(executor.submit(sum, [1,2,3,4,]))
            #results.append(executor.submit(preprocessing.one_block, (data, snum, session, raw, block_in_raw, block_in_experiment)))
            print '.'

for result in as_completed(results):
    print result
