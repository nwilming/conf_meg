from conf_analysis.meg import preprocessing
import sys


def
data = empirical.load_data()
data = empirical.data_cleanup(data)

for snum in range(15):
    for block in range(5):

        preprocessing.one_block(data, snum, session, raw, block_in_raw, block_in_experiment)
