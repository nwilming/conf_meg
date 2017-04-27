# -*- coding: utf-8 -*-
'''
Define epochs for the confidence data.
'''
import pandas as pd
import numpy as np

trigger_mapping = {150:'start', 151:'end', 64:'onset',
                   50:'contrast_change', 49:'stim_off', 48:'decision_start',
                   40:'ref_correct', 41:'stim_correct', 24:'RSP 1st Conf High',
                   23:'RSP 1st Conf Low', 22:'RSP 2nd Conf Low', 21:'RSP 2nd Conf High',
                   11:'Correct feedback', 10:'Incorrect feedback', 88:'no decision'}
triggers = dict((v, k) for k, v in trigger_mapping.items())


class Trial(object):

    def __init__(self):
        self.trial_stack = [(41,40), (150,),
                        (64,), (49,),
                        (64,), (50,), (50,), (50,), (50,), (50,), (50,), (50,), (50,), (50,), (48,),
                        (24, 23, 22, 21, 88),
                        (10, 11),
                        (151,)]
        self.field_names = ['correct_t', 'start',
                            'ref onset', 'ref offset',
                            'stim onset', 'cc1', 'cc2', 'cc3','cc4', 'cc5', 'cc6','cc7', 'cc8', 'cc9', 'stim offset',
                            'response_t',
                            'feedback_t',
                            'end']
        self.field2val = {'correct_t': lambda x: self.data.update({'correct':{41:'stim', 40:'ref'}[x]}),
                          'response_t': lambda x: (
                                                self.data.update(
                                                    {'response':{24:'ref', 23:'ref', 22:'stim', 21:'stim', 88:'error'}[x]}),
                                                self.data.update(
                                                    {'confidence':{24:'high', 21:'high', 22:'low', 23:'low', 88:'error'}[x]})
                                                ),
                         'feedback_t': lambda x: self.data.update({'feedback':{11:'F correct', 10:'F wrong'}[x]})}

        assert len(self.field_names) == len(self.trial_stack)
        self.data = {}

    def _field2val(self, value):
        pass


    def next(self, sample, value):
        expected = self.trial_stack.pop(0)
        field_name = self.field_names.pop(0)

        if value in expected:
            if field_name in self.field2val:
                self.field2val[field_name](value)
            if field_name == 'response_t' and value == triggers['no decision']:
                print('No decision:', value)
                self.trial_stack = []
                self.data[self.field_names.pop(0)] = np.nan
                self.data[self.field_names.pop(0)] = np.nan
                print(self.field_names)
            self.data[field_name] = sample

        else:
            print('Error trial:', self.data)
            print('Expecting: ', expected, 'Got:', value)
            print('Upcoming:', self.trial_stack)
            raise RuntimeError('Could not parse trial')
        if len(self.trial_stack) == 0 and len(self.field_names) ==0:
            raise StopIteration('Done')


def get_simpler(events):
    pass

def get_trials(events):
    '''
    Returns start and end times for trials and parses some features about them

    Push events into a reader and eventually get a trial back.
    '''
    trial_list = []

    trial = Trial()
    for start, _, value in events:
        try:
            trial.next(start, value)
        except StopIteration:
            trial_list.append(trial.data)
            trial = Trial()
    return pd.DataFrame(trial_list)



def get_marker(events, field='response_t', eid=0, baseline='start'):
    '''
    Returns start and end of the time period where the dynamic stimulus is shown.
    '''
    onsets = events.loc[:, field].values
    return np.vstack([onsets, np.zeros(onsets.shape), eid*np.ones(onsets.shape)]).T
