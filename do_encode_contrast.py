from conf_analysis.meg import tfr_analysis
import pandas as pd
import numpy as np
import patsy
import pyglmnet
from sklearn import cross_validation, svm, pipeline, preprocessing as skpre
from pymeg import decoding
from sklearn.linear_model import lasso_path, enet_path, LinearRegression
from sklearn.metrics import r2_score
from sklearn import decomposition
from sklearn import svm


def clf():
    return pipeline.Pipeline([
            ('scale', skpre.StandardScaler()),
            ('PCA', decomposition.PCA(n_components=.99)),
             ('encode', LinearRegression())])


def encode_mean_contrast(avg, meta, classifier=clf, cv=decoding.cv):
    # Need to buiuld y and X
    y = np.vstack(meta.contrast_probe.values).mean(1)
    y = (y-y.mean())/y.std()
    X = avg.unstack('freq')
    X = X.values

    idnan = np.isnan(y) | np.isnan(X).sum(1)>0
    y = y[~idnan]
    X = X[~idnan, :]
    results = []
    for i, (train_indices, test_indices) in enumerate(cv.split(y)):
        fold_result = {}
        np.random.shuffle(train_indices)
        fold = []
        clf = classifier()
        train = X[train_indices, :]
        fit = clf.fit(train, y[train_indices])
        test = X[test_indices, :]

        yh = fit.predict(X[test_indices, :])
        ytrain = fit.predict(X[train_indices,:])
        fold_result['predicted'] = yh
        fold_result['true'] = y[test_indices]
        fold_result['fold'] = i
        fold_result['r2'] = r2_score(y[test_indices], yh)
        print r2_score(y[train_indices], ytrain)
        results.append(fold_result)
    return pd.DataFrame(results)
