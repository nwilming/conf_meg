'''
Keep track of all the subject data
'''
from pylab import *
import os, socket
import pandas as pd


if socket.gethostname().startswith('node'):
    raw_path = '/home/nwilming/conf_data/raw/'
    downsampled = '/home/nwilming/conf_data/'
    convert_path = '/home/nwilming/MNE-2.7.0-3106-Linux-x86_64/bin/mne_ctf2fiff'
    cachedir = '/home/nwilming/conf_data/cache/'
else:
    raw_path = '/Volumes/dump/conf_data/raw/'
    downsampled = '/Volumes/dump/conf_data/'
    convert_path = '/Applications/MNE-2.7.4-3378-MacOSX-x86_64/bin/mne_ctf2fiff'
    cachedir = '/Users/nwilming/u/conf_analysis/cache/'

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
        dif = len(start)-len(end)
        # Aborted block during a trial, find location where [start start end] occurs
        start = where(events[:,2] == 150)[0]
        end = where(events[:, 2] == 151)[0]
        # TODO fix starts with end or end with start
        for i, (ts, te) in enumerate(zip(start[:-1], end)):
            if events[start[i+1],0] < events[te,0]:
                events[ts:start[i+1], :] = np.nan
                break
        events = events[~isnan(events[:,0]),:]
        start = where(events[:,2] == 150)[0]
        end = where(events[:, 2] == 151)[0]
        if dif == (len(start)-len(end)):
            raise RuntimeError('Something is wrong in the trial def. Fix this!')
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


admissible = [
    (41,40), (30, 31, 32), (64,), (49,),
    (64,), (50,), (50,), (50,), (50,), (50,), (50,), (50,), (50,), (50,), (48,),
    (24, 23, 22, 21, 88),
    (10, 11)]
field_names = [
    'meg_side', 'meg_noise_sigma', 'ref_onset', 'ref_offset',
    'stim_onset', 'cc0', 'cc1', 'cc2', 'cc3','cc4', 'cc5', 'cc6','cc7', 'cc8', 'cc9', 'stim_offset',
    'button', 'meg_feedback'
    ]

def fname2session(filename):
    #'/Volumes/dump/conf_data/raw/s04-04_Confidence_20151217_02.ds'
    return int(filename.split('/')[-1].split('_')[-2])

def get_meta(events, tstart, tend, tnum, bnum, day, subject):
    trls = []
    for ts, te, trialnum, block in zip(tstart, tend, tnum, bnum):
        trig_idx = (ts < events[:, 0]) & (events[:, 0] < te)
        trigs = events[trig_idx, :]
        trial = dict((k, v) for v, k in zip(trigs[:, 2], field_names))
        trial.update(dict((k+'_t', v) for v, k in zip(trigs[:, 0], field_names)))
        trial['trial'] = trialnum-1
        trial['block_num'] = block-1
        trial['start'] = ts
        trial['end'] = te
        trial['day'] = day
        trial['snum'] = subject
        trls.append(trial)

    trls = pd.DataFrame(trls)
    return trls


def correct_recording_errors(df):
    '''
    Cleanup some of the messy things that occured.

    S04-04: /Volumes/dump/conf_data/raw/s04-04_Confidence_20151217_02.ds
        - 6 blocks because of error in beginning. First two trials / first block needs to be killed
          : '/Volumes/dump/conf_data/raw/s04-03_Confidence_20151215_02.ds'
        - 7 blocks.
    '''
    if all((df.snum==4) & (df.day==20151217)):
        df = df.query('block_num>0')
        df.loc[:, 'block_num'] -= 1
    if all((df.snum==4) & (df.day==20151215)):
        df = df.query('block_num>1')
        df.loc[:, 'block_num'] -= 2
    return df


def cleanup(meta):
    '''
    A meta dataframe contains many duplicate columns. This function removes these
    and checks that no information is lost.
    '''
    cols = []
    # Check button + conf + response
    no_lates = meta.query('~(button==88)')
    assert all(isnan(no_lates.button)==isnan(no_lates.response))
    assert all(no_lates.button.replace({21:1, 22:1, 23:-1, 24:-1})==no_lates.response)
    assert all(no_lates.button.replace({21:2, 22:1, 23:1, 24:2})==no_lates.confidence)
    cols += ['button']
    assert all( (no_lates.meg_feedback-10)==no_lates.correct)
    cols += ['meg_feedback']
    assert all( (no_lates.meg_side.replace({40:-1, 41:1}))==no_lates.side)
    cols += ['meg_side']
    assert all( (no_lates.meg_noise_sigma.replace({31:.05, 32:.1, 33:.15}))==no_lates.noise_sigma)
    cols += ['meg_noise_sigma']
    cols += ['cc%i'%c for c in range(10)]
    cols += ['meg_side_t', ]
    return meta.drop(cols, axis=1)

def mne_events(data, time_field, event_val):
    return vstack([data[time_field], 0*data[time_field], data[event_val]]).astype(int).T
