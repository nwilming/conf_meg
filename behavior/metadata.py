'''
Keep track of all the subject data
'''
from pylab import *
import os, socket
import pandas as pd
from conf_analysis.meg.tools import deprecated

if socket.gethostname().startswith('node'):
    raw_path = '/home/nwilming/conf_meg/raw/'
    downsampled = '/home/nwilming/conf_data/'
    convert_path = '/home/nwilming/MNE-2.7.0-3106-Linux-x86_64/bin/mne_ctf2fiff'
    cachedir = '/home/nwilming/conf_data/cache/'
    behavioral_path = '/home/nwilming/conf_data/'

else:
    raw_path = '/Volumes/dump/conf_data/raw/'
    downsampled = '/Volumes/dump/conf_data/'
    convert_path = '/Applications/MNE-2.7.4-3378-MacOSX-x86_64/bin/mne_ctf2fiff'
    cachedir = '/Users/nwilming/u/conf_analysis/cache/'
    behavioral_path = '/Users/nwilming/u/conf_data/'


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

@deprecated
def get_datum_filename(sub, session, block):
    return os.path.join(raw_path, sub, '%s_sess%02i_block%02i.fif.gz'%(sub, session, block))


def define_blocks(events):
    '''
    Parse block structure from events in MEG files.
    '''
    events = events.astype(float)
    start = [0,0,]
    end = [0,]
    if not len(start) == len(end):
        dif = len(start)-len(end)
        start = where(events[:,2] == 150)[0]
        end = where(events[:, 2] == 151)[0]
        plot(events[end, 0], events[end, 0]*0, 'ko')
        plot(events[start, 0], events[start, 0]*0, 'rp')
        # Aborted block during a trial, find location where [start ... start end] occurs
        i_start, i_end = 0, 0   # i_start points to the beginning of the current
                                # trial and i_end to the beginning of the current trial.

        if len(start) > len(end):
            id_keep = (0*events[:,0]).astype(bool)
            start_times = events[start, 0]
            end_times = events[end, 0]
            for i, e in enumerate(end_times):
                d = start_times-e
                d[d>0] = -inf
                matching_start = argmax(d)
                evstart = start[matching_start]

                if (151 in events[evstart-10:evstart, 2]):
                    prev_end = 10-where(events[evstart-10:evstart, 2]==151)[0][0]
                    id_keep[(start[matching_start]-prev_end+1):end[i]+1] = True
                else:
                    id_keep[(start[matching_start]-10):end[i]+1] = True
            events = events[id_keep,:]

        start = where(events[:,2] == 150)[0]
        end = where(events[:, 2] == 151)[0]
        #plot(events[end, 0], events[end, 0]*0+1, 'ko')
        #plot(events[start, 0], events[start, 0]*0+1, 'rp')
        print len(start), len(end)
        if not (len(start)==len(end)):
            raise RuntimeError('Something is wrong in the trial def. Fix this!')


    trials = []
    blocks = []
    block = -1
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

val2field = {
    41:'meg_side', 40:'meg_side',
    31:'meg_noise_sigma', 32:'meg_noise_sigma', 33:'meg_noise_sigma',
    64:'_onset', 49:'ref_offset', 50:'cc', 48:'stim_offset',
    24:'button', 23:'button', 22:'button', 21:'button', 88:'button',
    10:'meg_feedback', 11:'meg_feedback'
    }


def fname2session(filename):
    #'/Volumes/dump/conf_data/raw/s04-04_Confidence_20151217_02.ds'
    return int(filename.split('/')[-1].split('_')[-2])

def get_meta(events, tstart, tend, tnum, bnum, day, subject):
    trls = []
    for ts, te, trialnum, block in zip(tstart, tend, tnum, bnum):
        trig_idx = (ts < events[:, 0]) & (events[:, 0] < te)
        trigs = events[trig_idx, :]
        trial = {}
        stim_state = ['stim', 'ref']
        cc_state = range(10)[::-1]
        for i, (v, t) in enumerate(zip(trigs[:, 2], trigs[:, 0])):
            fname = val2field[v]
            if v == 64:
                fname = stim_state.pop() + fname
            if v == 50:
                fname = fname + str(cc_state.pop())
            trial[fname] = v
            trial[fname + '_t'] = t

        trial['trial'] = trialnum-1
        trial['block_num'] = block
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
    #  S3 had a response fuckup where the participant had memorized a wrong key
    # mapping. This will be treated seperatedly.
    if 3 in unique(df.snum):
        id_button_21 = ((df.snum==3) & (df.button == 21) &
                      ((df.day == 20151207) | (df.day == 20151208)))
        id_button_22 =  ((df.snum==3) & (df.button == 22) &
                      ((df.day == 20151207) | (df.day == 20151208)))
        df.loc[id_button_21, 'button'] = 22
        df.loc[id_button_22, 'button'] = 21
    return df


def cleanup(meta):
    '''
    A meta dataframe contains many duplicate columns. This function removes these
    and checks that no information is lost.
    '''
    cols = []
    # Check button + conf + response
    no_lates = meta.loc[~isnan(meta.button)]
    no_lates = no_lates.query('~(button==88)')

    # Check for proper button to response mappings!
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
