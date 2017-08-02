
from conf_analysis.meg import tfr_analysis
import pandas as pd
import numpy as np
import patsy
import pyglmnet
from sklearn import cross_validation, svm, pipeline, preprocessing as skpre
from pymeg import decoding
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn import decomposition
from sklearn import svm
import pylab as plt

def clf():
    return pipeline.Pipeline([
            ('scale', skpre.StandardScaler()),
            ('PCA', decomposition.PCA(n_components=250)),
            ('encode', svm.SVR())])


def encode_mean_contrast(avg, meta, classifier=clf, cv=decoding.cv):
    # Need to buiuld y and X
    y = np.vstack(meta.contrast_probe.values)[:, 3]
    y = (y-y.mean())/y.std()
    #X = avg.unstack('freq')
    X = avg#X.values

    idnan = np.isnan(y) | np.isnan(X).sum(1)>0
    y = y[~idnan]
    X = X[~idnan, :]
    results = []
    for i, (train_indices, test_indices) in enumerate(cv.split(y)):
        X_train, X_test = X[train_indices, :], X[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]

        fold_result = {}

        fold = []
        clf = classifier()
        fit = clf.fit(X_train, y_train)
        yh = fit.predict(X_test)

        plt.plot(y_test, yh, 'o')
        ytrain = fit.predict(X_train)
        fold_result['predicted'] = yh
        fold_result['true'] = y_test
        fold_result['fold'] = i
        fold_result['r2'] = r2_score(y_test, yh)
        print np.corrcoef(y_train, ytrain)[0,1], np.corrcoef(y_test, yh)[0,1]
        results.append(fold_result)
    return pd.DataFrame(results)
