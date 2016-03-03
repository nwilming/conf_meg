'''
Keep track of all the subject data
'''
from pylab import *
import os, socket

if socket.gethostname().startswith('node'):
    raw_path = '/home/nwilming/conf_data/raw/'
    data_path = '/home/nwilming/conf_data/'
    convert_path = '/home/nwilming/MNE-2.7.0-3106-Linux-x86_64/bin/mne_ctf2fiff'

else:
    raw_path = '/Volumes/dump/conf_data/raw/'
    data_path = '/Volumes/dump/conf_data/'
    convert_path = '/Applications/MNE-2.7.4-3378-MacOSX-x86_64/bin/mne_ctf2fiff'


data_files = {'S01': ['s01-01_Confidence_20151208_02.ds',
                      's01-02_Confidence_20151210_02.ds',
                      's01-03_Confidence_20151216_01.ds',
                      's01-04_Confidence_20151217_02.ds'],
              'S02': ['s02-01_Confidence_20151208_02.ds',
                      's02-02_Confidence_20151210_02.ds',
                      's02-03_Confidence_20151211_01.ds',
                      's02-4_Confidence_20151215_02.ds'],
              'S03': ['s03-01_Confidence_20151208_01.ds',
                      's03-02_Confidence_20151215_02.ds',
                      's03-03_Confidence_20151216_02.ds',
                      's03-04_Confidence_20151217_02.ds'],
              'S04': ['s04-01_Confidence_20151210_02.ds',
                      's04-02_Confidence_20151211_02.ds',
                      's04-03_Confidence_20151215_02.ds',
                      's04-04_Confidence_20151217_02.ds'],
              'S05': ['s05-01_Confidence_20151210_02.ds',
                      's05-02_Confidence_20151211_02.ds',
                      's05-03_Confidence_20151216_02.ds',
                      's05-04_Confidence_20151217_02.ds'],
              'S06': ['s06-01_Confidence_20151215_02.ds',
                      's06-02_Confidence_20151216_02.ds',
                      's06-03_Confidence_20151217_02.ds',
                      's06-04_Confidence_20151218_04.ds']}


def get_fif_filenames(first):
    files = glob.glob(first.replace('raw.fif', '*.fif'))
    return files

def get_sub_session_rawname(sub, session):
    return os.path.join(raw_path, data_files[sub][session])

def get_datum_filename(sub, session, block):
    return os.path.join(data_path, sub, '%s_sess%02i_block%02i.fif.gz'%(sub, session, block))


def define_blocks(events):
    '''
    Parse block structure from events in MEG files.
    '''
    print events.shape
    start = [0,0,]
    end = [0,]
    while not len(start) == len(end):
        # Aborted block during a trial, find location where [start start end] occurs
        start = where(events[:,2] == 150)[0]
        end = where(events[:, 2] == 151)[0]
        # TODO fix starts with end or end with start
        for i, (ts, te) in enumerate(zip(start[:-1], end)):
            if events[start[i+1],0] < events[te,0]:
                events[ts:start[i+1], :] = np.nan
                break
        events = events[~isnan(events[:,0]),:]

    trials = []
    blocks = []
    block = 0
    for i, (ts, te) in enumerate(zip(start, end)):
        # Get events just before trial onset, they mark trial numbers
        trial_nums = events[ts-8:ts+1, 2]
        pins = trial_nums[(0<=trial_nums) & (trial_nums<=8)]
        if len(pins) == 0:
            trial = 1
        else:
            # Convert pins to numbers
            trial = 0
            trial = sum([2**(8-pin) for pin in pins])
        if trial == 1:
            block += 1
        trials.append(trial)
        blocks.append(block)
    return events[start,0], events[end, 0], np.array(trials), np.array(blocks)
