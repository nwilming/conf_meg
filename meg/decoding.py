'''
Do some decoding stuff.
'''
from conf_analysis.meg import preprocessing
from sklearn import cross_validation, svm, pipeline, preprocessing as skpre
from sklearn import decomposition
import numpy as np
import pandas as pd
from joblib import Memory

memory = Memory(cachedir=metadata.cachedir)

clf = lambda: pipeline.Pipeline([
            ('scale', skpre.StandardScaler()),
            ('PCA', decomposition.PCA(n_components=51)),
            ('SVM', svm.LinearSVC())])

cv = lambda x: cross_validation.StratifiedShuffleSplit(x, n_iter=10, test_size=0.1)


@memory.cache
def decode(classifier, data, labels, train_time, predict_times,
        cv=cross_validation.StratifiedKFold):
    '''
    Apply a classifier to data and predict labels with cross validation.
    Train classifier from data at data[:, :, train_time] and apply to all
    indices in predict_times. Indices are interpreted as an index into
    the data matrix.

    Returns a vector of average accuracies for all indices in predict_idx.

    Parameters
    ----------
    classifier : sklearn classifier object
    data : np.array, (#trials, #features, #time)
    labels : np.array, (#trials,)
    train_time : int
    predict_times : iterable of ints
    '''
    assert len(labels) == data.shape[0]
    results = []

    for i, (train_indices, test_indices) in enumerate(cv(labels)):
        np.random.shuffle(train_indices)
        fold = []
        clf = classifier()
        l1, l2 = np.unique(labels)
        l1 = train_indices[labels[train_indices]==l1]
        l2 = train_indices[labels[train_indices]==l2]
        if len(l1)>len(l2):
            l1 = l1[:len(l2)]
        else:
            l2 = l2[:len(l1)]
        assert not any([k in l2 for k in l1])
        train_indices = np.concatenate([l1, l2])
        train = data[train_indices, :, train_time]
        clf.fit(train, labels[train_indices])

        for pt in predict_times:
            results.append({
                'fold':i,
                'train_time':train_time,
                'predict_time':pt,
                'accuracy': clf.score(data[test_indices, :, pt], labels[test_indices])})
    return pd.DataFrame(results)


def generalization_matrix(epochs, labels, dt, classifier=clf, cv=cv):
    '''
    Get data for a generalization across time matrix.

    Parameters
    ----------
        epochs: mne.epochs object
    Epochs to use for decoding.
        labels : np.array (n_epochs,)
    Target labels to predict
        dt : int
    Time resolution of the decoding in ms (!).
    '''
    data = epochs._data
    sfreq = epochs.info['sfreq']

    tlen = data.shape[-1]/(float(sfreq)/1000.)
    nsteps = around(float(tlen)/dt)
    steps = linspace(0, data.shape[-1], nsteps)

    r = pd.concat(
        [decoding.decode(clf, data, labels, int(tt), steps, cv=cv)
            for tt in steps]
        )
    return r


def apply_decoder(func, snum, epoch, label):
    '''
    Apply a decoder function to epochs from a subject and decode 'label'.

    Parameters
    ----------
        func: function object
    A function that performs the desired decoding. It needs to take two arguments
    that encode data and labels to use for the decoing. E.g.:

        >>> func = lambda x,y: generalization_matrix(x, y, 10)

        snum: int
    Subject number to indicate which data to load.
        epoch: str
    One of 'stimulus', 'response', or 'feedback'
        label: str
    Which column in the metadata to use for decoding. Labels will recoded to
    0-(num_classes-1).
    '''
    s, m = preprocessing.get_epochs_for_subject(snum, epoch) #This will cache.
    s = s.pick_channels([m for m in s.ch_names if m.startswith('M')])

    # Drop nan labels
    nan_loc = m.index[isnan(m.loc[:, label])]
    use_loc = m.index[~isnan(m.loc[:, label])]

    m = m.drop(nan_loc)
    s = s[list(use_loc.astype(str))]

    # Sort order index to align epochs with labels.
    m = m.sort_index()

    # Recode labels to 0-(n-1)
    labels = m.loc[:, label]
    labels = skpre.LabelEncoder().fit(labels).transform(labels)
    return func(s._data, labels)
