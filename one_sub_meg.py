from conf_analysis.meg import artifacts, preprocessing, keymap
from conf_analysis.behavior import empirical, metadata
import mne, locale
import time
import numpy as np
import cPickle


start = time.time()

locale.setlocale(locale.LC_ALL, "en_US")

snum = 2

filenames = [metadata.get_sub_session_rawname('S%02i'%snum, x) for x in range(4)]

data = empirical.load_data()
data = empirical.data_cleanup(data)

# Compute a unique hash for all entries in data
def ukey(x):
    x = [str(v) for v in x]
    x = int(''.join(x))
    digits = int(math.log10(x))+1
    return x-2015*10**(digits-3)

data.loc[:, 'hash'] =  [keymap.hash(x) for x in data.loc[:, ('day', 'snum', 'block_num', 'trial')].values]

assert len(np.unique(data.loc[:, 'hash'])) == len(data.loc[:, 'hash'])



for i, filename in enumerate(filenames[1:]):
    raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
    trials = preprocessing.blocks(raw)

    assert len(trials['block'])==500
    assert len(np.unique(trials['block']))==5
    assert sum(trials['trial']) == sum(range(1,101))*5

    block_cnt = 0
    for block in np.unique(trials['block']):
        start_iter = time.time()
        # Load data and preprocess it.
        r, r_id = preprocessing.load_block(raw, trials, block)
        r, ants, artdefs = preprocessing.preprocess_block(r)
        artdefs['id'] = r_id

        meta, timing = preprocessing.get_meta(data, r, snum, block_cnt)

        outname = '/Volumes/dump/conf_data/S%i/S%02i_S%i_B%i'%(snum, snum, i, block_cnt)
        cPickle.dump(artdefs, open(outname + '.artifacts', 'w'), protocol=2)

        # Define stimulus epochs.
        m, s = preprocessing.get_epoch(r, meta, timing,
                                       event='stim_onset_t', epoch_time=(-.2, 1.5),
                                       base_event='stim_onset_t', base_time=(-.2, 0))
        print 'Dropped ', s.drop_log_stats(), 'epochs'
        if len(s)>0:
            s = s.resample(600, npad='auto')
            s.save(outname + 'stim_epoch-epo.fif.gz')
            m.to_hdf(outname + 'stim_epoch.meta', 'meta')

        # Define response epochs.
        m, s = preprocessing.get_epoch(r, meta, timing,
                                       event='button_t', epoch_time=(-1.5, .5),
                                       base_event='stim_onset_t', base_time=(-.2, 0))

        print 'Dropped ', s.drop_log_stats(), 'epochs'
        if len(s)>0:
            s = s.resample(600, npad='auto')
            s.save(outname + 'response_epoch-epo.fif.gz')
            m.to_hdf(outname + 'response_epoch.meta', 'meta')

        # Define feedback epochs
        m, s = preprocessing.get_epoch(r, meta, timing,
                                       event='meg_feedback_t', epoch_time=(-.5, .5),
                                       base_event='stim_onset_t', base_time=(-.2, 0))
        print 'Dropped ', s.drop_log_stats(), 'epochs'
        if len(s)>0:
            s = s.resample(600, npad='auto')
            s.save(outname + 'feedback_epoch-epo.fif.gz')
            m.to_hdf(outname + 'feedback_epoch.meta', 'meta')

        end_iter = time.time()
        block_cnt += 1
        print 'This session took:', round(end_iter-start_iter, 2), 's'

print 'This subject took:', round(time.time()-start, 2), 's'
