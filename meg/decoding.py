'''
Do some decoding stuff.
'''
from sklearn import cross_validation, svm, pipeline, preprocessing as skpre
from sklearn import decomposition
import numpy as np


clf = lambda: pipeline.Pipeline([
            ('scale', skpre.StandardScaler()),
            ('PCA', decomposition.PCA(n_components=150)),
            ('SVM', svm.LinearSVC())])

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

    results = []

    for train_indices, test_indices in cv(labels):
        print test_indices
        fold = []
        clf = classifier()
        train = data[train_indices, :, train_time]
        clf.fit(train, labels[train_indices])
        print sum(labels[test_indices]), len(test_indices)
        for pt in predict_times:
            fold += [clf.score(data[test_indices, :, pt], labels[test_indices])]
        results.append(fold)
    results = np.array(results).mean(0)
    return results
