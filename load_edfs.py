import numpy as np
import pupil
import pandas as pd

def listfiles(dir):
    import glob, os, time
    files = glob.glob(os.path.join(dir, '*.edf'))
    data = {}
    subs = []
    for f in files:
        if 'localizer' in f:
            continue
        sub, d = f.replace('.edf', '').split('/')[-1].split('_')
        data[time.strptime(d, '%Y%m%dT%H%M%S')] = f
        subs.append(sub)
    assert(len(np.unique(subs))==1)
    return data, sub


def save_as_df(files, sub):
    es = []
    msgs = []
    keys = []
    for i, f in enumerate(files):
        session = (i/5) +1
        block = np.mod(i, 5) +1
        print '\nProcessing S%i, B%i: '%(session, block), f
        events, messages = pupil.load_edf(f)
        events, messages = pupil.preprocess(events, messages)
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
