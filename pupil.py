# -*- coding: utf-8 -*-
'''
Load EDFs and prepare data frames.
'''
import collections
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from pylab import *
import sympy
import patsy
import seaborn as sns
sns.set_style('ticks')

from pyedfread import edf

EDF_HZ = 1000.0

def expand(events, messages, field, align_key='trial', align_time='trial_time', default=nan):
    '''
    --- Missing ---
    '''
    assert(len(unique(events[align_key])) == 1)
    trial = events[align_key].iloc[0]
    messages = messages[messages[align_key] == trial]
    try:
        con = messages[field].iloc[0]
        con_on = messages[align_time].iloc[0]
    except IndexError:
        print trial
        events[field] =  default + zeros(len(events))
        return events

    trial_begin = events.index.values[0]
    events[field] = default + zeros(len(events))
    for (start, end), value in zip(zip(con_on[:-1], con_on[1:]), con):

        events[field].loc[start:end] = value
    events[field].loc[end:end+100] = con[-1]

    return events


def load_behavior(behavioral):
    def unbox_messages(current):
        for key in current.keys():
            try:
                if len(current[key])==1:
                    current[key] = current[key][0]
            except TypeError:
                pass
        return current
    # Also load behavioral results file to match with contrast levels shown
    behavioral = loadmat(behavioral)['session_struct']['results'][0,0]
    d = []
    fields = behavioral.dtype.fields.keys()
    for trial in range(behavioral.shape[1]):
        d.append({})
        for field in fields:
            d[-1][field] = behavioral[0, trial][field].ravel()
        d[-1]['trial'] = trial+1
        d[-1] = unbox_messages(d[-1])
    return pd.DataFrame(d)


def load_edf(filename):
    '''
    Loads one EDF file and returns a clean DataFrame.
    '''
    events, messages = edf.pread(
        filename,
        properties_filter=['gx', 'gy', 'pa', 'sttime', 'start'],
        filter='all')

    #events = edf.trials2events(events, messages)
    #events['stime'] = pd.to_datetime(events.sample_time, unit='ms')
    interp_blinks(events, 100, 100, 100, ['left_pa', 'left_gx', 'left_gy'])
    if all(events.right_pa == -32768):
        del events['right_pa']
        events['pa'] = events.left_pa
        del events['left_pa']
    elif all(events.left_pa == -32768):
        del events['left_pa']
        events['pa'] = events.right_pa
        del events['right_pa']
    else:
        raise RuntimeError('Recorded both eyes? So unusual that I\'ll stop here')

    # In some cases the decision variable still contains ['second', 'conf', 'high'], 21.0 -> Fix this
    # In these cases the decision_time variable has 2 time stamps as well...
    if messages.decision.dtype == dtype(object):
        messages['decision'] = array([x[-1] if isinstance(x, collections.Sequence) else x for x in messages.decision.values])
        messages['decision_time'] = array([x[-1] if isinstance(x, collections.Sequence) else x for x in messages.decision_time.values])

    return events, messages


def join_edf_and_behavior(messages, behavior):
    return messages.set_index('trial').join(behavior.set_index('trial')).reset_index()


def preprocess(events, messages, behavior=None):
    if behavior is not None:
        messages = join_edf_and_behavior(messages, behavior)
    events = events.set_index('sample_time')
    events = events.groupby('trial').apply(lambda x: expand(x, messages, 'contrast_probe',
                                                            align_time='con_change_time', default=0))
    events = decimate(events, 10)
    events = events[~isnan(events.pa)]
    #con_on = [c.conrast_time.values[0][0] for name, c in messages.groupby('trial')]
    #messages['contrast_on'] = con_on
    join_msg(events, messages)
    below, filt, above = filter_pupil(events.pa, 100)
    events['pafilt'] = filt
    events['palow'] = below
    events['pahigh'] = above
    return events, messages


def join_msg(events, messages):
    # Join messages into events. How to do this depends a bit on the semantics of the message
    paired_fields = ['decision', 'feedback']
    time_index = events.index.values
    for field in paired_fields:
        events[field] = zeros(events.pa.values.shape)
        for t,v in zip(messages[field + '_time'].values, messages[field].values):
            idx = argmin(abs(time_index-t))
            events[field].iloc[idx] = v


def interp_blinks(events, pre, post, offset=10, fields=['left_pa']):
    '''
    Linearly interpolate blinks
    '''
    d = diff(events.blink)
    blinks = where(d)[0].tolist()
    if events.blink.values[0] == 1:
        #Starts with a blink that is missing in blinks. Add it
        blinks.insert(0, 0)
    if events.blink.values[-1] == 1:
        #Ends with blink that will not be detected. Add it
        blinks.append(len(events.blink.values))
    assert(mod(len(blinks), 2)==0)

    for start, end in zip(blinks[0::2], blinks[1::2]):
        for field in fields:
            before = nanmean(events[field].values[start-pre:start-offset])
            after = nanmean(events[field].values[end:end+start+offset])
            filler = before + arange(end-start)* (after-before)/float(end-start)
            events[field].values[start:end] = filler


def make_design_matrix(events):
    '''
    Make a design matrix for regression.
    '''
    pass


def decimate(data, factor, **kwargs):
    '''
    Donwsample a data frame by downsampling all columns.
    Forces the use of FIR filter to avoid a phase shift.
    Removes the firs factor samples to avoid edge artifacts in the beginning.
    '''
    target = {}
    kwargs['ftype'] = 'fir'
    for column in data.columns:
        target[column] = signal.decimate(data[column], factor, **kwargs)[factor:]
    index = signal.decimate(data.index.values, factor, **kwargs)[factor:]
    return pd.DataFrame(target, index=index)


def filter_pupil(pupil, sampling_rate, highcut = 10., lowcut = 0.01):
    """
    Band pass filter using a butterworth filter of order 3.

    lowcut: Cut off everything slower than this
    highcut: Cut off everything that is faster than this

    Returns:
        below: everything below the passband
        filtered: everything in the passband
        above: everything above the passband

    Based on scipy cookbook: https://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
    """

    def butter_bandpass(lowcut, highcut, fs, order=5):
       nyq = 0.5 * fs
       low = lowcut / nyq
       high = highcut / nyq
       b, a = signal.butter(order, [low, high], btype='band')
       return b, a

    b, a = butter_bandpass(lowcut, highcut, sampling_rate, 3)
    filt_pupil = signal.filtfilt(b, a, pupil)

    b, a = butter_bandpass(highcut, 0.5*sampling_rate-0.5, sampling_rate, 3)
    above = signal.filtfilt(b, a, pupil)
    return pupil - (filt_pupil+above), filt_pupil, above

def eval_model(model, data):
    import patsy_transforms as pt
    from sklearn import linear_model
    import statsmodels.api as sm
    y,X = patsy.dmatrices(model, data=data.copy(), eval_env=1)
    m = linear_model.LinearRegression()
    idnan = isnan(y.ravel())
    mod = sm.OLS(y[~idnan, :], X[~idnan, :])
    res = mod.fit()
    print res.summary(xname=X.design_info.column_names)
    m.fit(X[~idnan,:],y[~idnan,:])
    yh = m.predict(X)
    print corrcoef(y.ravel(), yh.ravel())[0,1]**2
    return m, yh, y, X, res



def IRF_pupil(fs=100, dur=4, s=1.0/(10**26), n=10.1, tmax=.930):
    """
    Canocial pupil impulse fucntion [/from JW]

    dur: length in s

    """

    # parameters:
    timepoints = np.linspace(0, dur, dur*fs)

    # sympy variable:
    t = sympy.Symbol('t')

    # function:
    y = ( (s) * (t**n) * (math.e**((-n*t)/tmax)) )

    # derivative:
    y_dt = y.diff(t)

    # lambdify:
    y = sympy.lambdify(t, y, "numpy")
    y_dt = sympy.lambdify(t, y_dt, "numpy")

    # evaluate and normalize:
    y = y(timepoints)
    y = y/np.std(y)
    y_dt = y_dt(timepoints)
    y_dt = y_dt/np.std(y_dt)

    # dispersion:
    y_dn = ( (s) * (timepoints**(n-0.01)) * (math.e**((-(n-0.01)*timepoints)/tmax)) )
    y_dn = y_dn / np.std(y_dn)
    y_dn = y - y_dn
    y_dn = y_dn / np.std(y_dn)

    return y, y_dt, y_dn
