'''
Preprocess an MEG data set.
'''
import mne
from pylab import *
import pandas as pd
import glob
from itertools import product
import metadata


def block(raw, bnum, block, ts, te):
    tstart = raw.index_as_time(ts[block==bnum].min()) - 5
    tend = raw.index_as_time(te[block==bnum].max()) + 5
    raw_block = raw.crop(tstart, tend, copy=True)
    events_buttons = mne.find_events(raw_block, 'UPPT002')
    events_triggers = mne.find_events(raw_block, 'UPPT001')
    eog_channels = ['UADC002-3705', 'UADC003-3705', 'UADC004-3705'] # Hor, Vert, pupil
    eog_events_2 = mne.preprocessing.find_eog_events(raw_block, event_id=1002, ch_name=str(eog_channels[0]))
    eog_events_3 = mne.preprocessing.find_eog_events(raw_block, event_id=1003, ch_name=str(eog_channels[1]))
    eog_events_4 = mne.preprocessing.find_eog_events(raw_block, event_id=1004, ch_name=str(eog_channels[2]))

    events = vstack((events_triggers, events_buttons, eog_events_2, eog_events_3, eog_events_4))
    raw_block = raw_block.load_data()
    return raw_block
    raw_block, events_resampled = raw_block.resample(500, events=events, copy=False, jobs=3)
    outfile = metadata.get_datum_filename(subject, session, bnum)
    raw_block.save(outfile)
    events = pd.DataFrame(events, columns=['sample', 'next', 'event_value'])
    cPickle.dump({'events_resampled': events_resampled, 'events':events, 'filename':filename}, open(filename.replace('fif.gz', 'events.pkl'), 'w'))
    del raw_block



def subject(subject):
    [session(subject, k) for k in range(4)]


def session(subject, session):
    filename = metadata.get_sub_session_rawname(subject, session)
    raw = mne.io.read_raw_ctf(filename, system_clock='ignore')
    events = mne.find_events(raw, 'UPPT001')
    ts, te, trial, block = metadata.define_blocks(events)
    blocks = unique(block)
    return raw, ts, te, trial, block, blocks, events
