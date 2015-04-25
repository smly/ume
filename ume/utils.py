# -*- coding: utf-8 -*-
import logging as l
import importlib
import os
import json

import numpy as np
import scipy.io as sio
from sklearn.cross_validation import KFold, ShuffleSplit, LeaveOneOut
from sklearn.externals.joblib import Parallel, delayed

import ume.externals.jsonnet


def feature_functions(module_name):
    somemodule = importlib.import_module(module_name)
    return somemodule, [
        name for name in somemodule.__dict__.keys()
        if name.startswith('gen_')
    ]


def dynamic_load(cls_path):
    parts = cls_path.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]
    somemodule = importlib.import_module(module_name)
    return getattr(somemodule, class_name)


MAT_FORMAT = "./data/working/{0}.mat"
NPZ_FORMAT = "./data/working/{0}.npz"

def save_npz(feature_name, output_dict):
    np.savez(NPZ_FORMAT.format(feature_name), **output_dict)


def save_mat(feature_name, output_dict):
    sio.savemat(MAT_FORMAT.format(feature_name), output_dict)


def load_mat(path):
    if path.endswith('npz'):
        return np.load(path)
    elif path.endswith('mat'):
        return sio.loadmat(path)
    else:
        raise RuntimeError("Unsupported filetype: npz or mat are required")


def get_feature_stats(func_name):
    exist_mat = os.path.exists(MAT_FORMAT.format(func_name))
    exist_npz = os.path.exists(NPZ_FORMAT.format(func_name))

    print(exist_mat, exist_npz, func_name)

    if exist_mat:
        return True, MAT_FORMAT.format(func_name)
    elif exist_npz:
        return True, NPZ_FORMAT.format(func_name)
    else:
        return False, None


class PredictForRegression(object):
    def __init__(self, settings):
        self.metrics = settings['metrics']
        self.prediction = settings['prediction']
        self.model = settings['model']

    def solve(self, X_train, X_test, y_train):
        params = self.prediction['params']
        if "dense" in params and params['dense'] == "True":
            l.info("Convert sparse matrix into dense matrix")
            X_train = X_train.todense()
            X_test = X_test.todense()
            l.info("Convert sparse matrix into dense matrix ... done")

        klass = dynamic_load(self.model['class'])
        clf = klass(**self.model['params'])
        l.info("Training model: {0}".format(str(clf)))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        return y_pred


class MultiClassPredictProba(object):
    def __init__(self, settings):
        self.metrics = settings['metrics']
        self.prediction = settings['prediction']
        self.model = settings['model']

    def solve(self, X_train, X_test, y_train):
        klass = dynamic_load(self.model['class'])
        clf = klass(**self.model['params'])
        l.info("Training model: {0}".format(str(clf)))
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)


class PredictProba(object):
    def __init__(self, settings):
        self.metrics = settings['metrics']
        self.prediction = settings['prediction']
        self.model = settings['model']

    def solve(self, X_train, X_test, y_train):
        params = self.prediction['params']
        if "dense" in params and params['dense'] == "True":
            l.info("Convert sparse matrix into dense matrix")
            X_train = X_train.todense()
            X_test = X_test.todense()
            l.info("Convert sparse matrix into dense matrix ... done")

        klass = dynamic_load(self.model['class'])
        clf = klass(**self.model['params'])
        l.info("Training model: {0}".format(str(clf)))
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 0]

        return y_pred


def load_settings(path):
    if path.endswith(".jsonnet") or path.endswith(".jn"):
        settings = ume.externals.jsonnet.load(path)
    else:
        with open(path, 'r') as f:
            settings = json.load(f)

    return settings


def _cv(X, y, idx_train, idx_test, settings):
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    metrics = dynamic_load(settings['metrics']['method'])
    metrics_params = settings['metrics']['params']
    prediction = settings['prediction']

    predict_klass = dynamic_load(prediction['method'])
    p = predict_klass(settings)
    y_pred = p.solve(X_train, X_test, y_train)

    score = metrics(y_test, y_pred, **metrics_params)
    l.info("Score: {0:.4f}".format(score))
    return score


def kfoldcv(X, y, settings):
    kfold_params = settings['cross_validation']['params']
    n_jobs = kfold_params.get('n_jobs', 1)
    n_folds = kfold_params.get('n_folds', 5)

    kf = KFold(X.shape[0], n_folds=n_folds, shuffle=True, random_state=777)
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_cv)(X, y, idx_train, idx_test, settings)
        for idx_train, idx_test in kf
    )

    return np.array(scores).mean(), np.array(scores).var()


def _load_features(f_names):
    X = None
    for f_name in f_names:
        l.info(f_name)
        var_name = 'X'
        if type(f_name) is dict:
            var_name = f_name['name']
            f_name = f_name['file']

        X_add = load_mat(f_name)[var_name]
        if X is None:
            X = X_add
        elif type(X) is np.ndarray and type(X_add) is np.ndarray:
            X = np.hstack((X, X_add))
        else:
            X = X_add if X is None else ss.hstack((X, X_add))
    return X


def load_dataset(settings):
    X = _load_features(settings['features'])
    idx_train = sio.loadmat(settings['idx']['train']['file'])[
        settings['idx']['train']['name']
    ]
    idx_test = sio.loadmat(settings['idx']['test']['file'])[
        settings['idx']['test']['name']
    ]
    idx_train = idx_train[:, 0]
    idx_test = idx_test[:, 0]
    X_train = X[idx_train]
    X_test = X[idx_test]
    y_train = sio.loadmat(settings['target']['file'])[
        settings['target']['name']
    ]
    #y_train = y_train[:, 0, 0]
    y_train = y_train[:, 0]
    return X_train, X_test, y_train


def loocv(X, y, settings):
    prediction = settings['prediction']

    scores = []
    l.info("Matrix shape: {0}".format(X.shape))
    loo = LeaveOneOut(X.shape[0])
    for idx_train, idx_test in loo:
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        predict_klass = dynamic_load(prediction['method'])
        p = predict_klass(settings)
        y_pred = p.solve(X_train, X_test, y_train)

        if y_pred.tolist()[0] >= 0.5 and y_test.tolist()[0] >= 0.5:
            scores.append(1.0)
        elif y_pred.tolist()[0] < 0.5 and y_test.tolist()[0] < 0.5:
            scores.append(1.0)
        else:
            scores.append(0.0)

        l.info("Score: yes={0}, trial={1}, size={2}".format(
            int(np.array(scores).sum()), len(scores), X.shape[0]))

    return np.array(scores).mean(), np.array(scores).var()


def unsafe_shuffle_split(X, y, settings):
    metrics = dynamic_load(settings['metrics']['method'])
    metrics_params = settings['metrics']['params']
    cv_params = settings['cross_validation']['params']
    prediction = settings['prediction']

    scores = []
    k = cv_params.get('n_iter', 5)

    while True:
        ss = ShuffleSplit(X.shape[0], **cv_params)
        for idx_train, idx_test in ss:
            X_train, X_test = X[idx_train], X[idx_test]
            y_train, y_test = y[idx_train], y[idx_test]

            if y_test.sum() < 1.0:
                continue

            predict_klass = dynamic_load(prediction['method'])
            p = predict_klass(settings)
            y_pred = p.solve(X_train, X_test, y_train)

            score = metrics(y_test, y_pred, **metrics_params)
            l.info("Score: {0:.4f}".format(score))
            scores.append(score)
            if len(scores) >= k:
                break
        if len(scores) >= k:
            break

    return np.array(scores).mean(), np.array(scores).var()
