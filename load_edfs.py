import numpy as np
import pupil
import pandas as pd

def listfiles(dir):
    import glob, os, time
    edffiles = glob.glob(os.path.join(dir, '*.edf'))
    matfiles = glob.glob(os.path.join(dir, '*.mat'))
    edfdata = {}
    matdata = {}
    subs = []
    for f in edffiles:
        if 'localizer' in f:
            continue
        sub, d = f.replace('.edf', '').split('/')[-1].split('_')
        edfdata[time.strptime(d, '%Y%m%dT%H%M%S')] = f
        subs.append(sub)
    for f in matfiles:
        if 'localizer' in f or 'quest' in f:
            continue
        sub, d, ll = f.replace('.mat', '').split('/')[-1].split('_')
        print sub, d
        matdata[time.strptime(d.replace('Dec', 'Dez'), '%d-%b-%Y %H:%M:%S')] = f
        subs.append(sub)
    assert(len(np.unique(subs))==1)
    return edfdata, matdata, sub


def save_as_df(sub, edffiles, matfiles):
    es = []
    msgs = []
    keys = []
    for i, (f, m) in enumerate(zip(edffiles, matfiles)):

        session = (i/5) +1
        block = np.mod(i, 5) +1
        print '\nProcessing S%i, B%i: '%(session, block), f
        events, messages = pupil.load_edf(f)
        behavior = pupil.load_behavior(m)
        events, messages = pupil.preprocess(events, messages, behavior)
        events['block'] = block
        events['session'] = session
        events['subject'] = sub
        messages['block'] = block
        messages['session'] = session
        messages['subject'] = sub
        es.append(events)
        msgs.append(messages)
        keys.append((session, block))
    events = pd.concat(es, keys=keys)
    messages = pd.concat(msgs, keys=keys)
    import cPickle
    cPickle.dump({'events':events, 'messages':messages}, open(sub+'.pickle', 'w'), protocol=2)
