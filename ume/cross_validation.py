# -*- coding: utf-8 -*-
import logging as l

import numpy as np
from sklearn.cross_validation import KFold, ShuffleSplit, LeaveOneOut
from sklearn.externals.joblib import Parallel, delayed

from ume.utils import dynamic_load


def _cv(ith, X, y, idx_train, idx_test, metrics, task_class):
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    y_pred = task_class.solve(X_train, y_train, X_test)
    score = metrics(y_test, y_pred)
    l.info("CV score: {2:.4f} ({0} of {1})".format(ith, X.shape[0], score))
    return score


def loocv(X, y, task_class, n_jobs=1):
    task_metrics = task_class._conf['task']['params']['metrics']
    task_method = task_metrics['method']
    metrics = dynamic_load(task_method)

    loo = LeaveOneOut(X.shape[0])
    cv_scores = Parallel(n_jobs=n_jobs)(
        delayed(_cv)(ith, X, y, idx_train, idx_test, metrics, task_class)
        for ith, (idx_train, idx_test) in enumerate(loo)
    )

    mean_cv_score = np.mean(cv_scores)
    l.info("CV Score: {0:.4f} (var: {1:.6f})".format(
        mean_cv_score,
        np.var(cv_scores)))

    return mean_cv_score


def kfold(X, y, task_class, n_folds=10, skip_all_zero_target=None):
    task_metrics = task_class._conf['task']['params']['metrics']
    task_method = task_metrics['method']
    metrics = dynamic_load(task_method)

    cv_scores = []
    #ss = SSS(y, 5, test_size=0.1, random_state=777)
    #for kth, (train_idx, test_idx) in enumerate(ss):
    #    X_train, X_test = X[train_idx], X[test_idx]
    #    y_train, y_test = y[train_idx], y[test_idx]
    #    y_pred = self.solve(X_train, y_train, X_test)
    #    score = metrics(y_test, y_pred)
    #    l.info("KFold: ({0}) {1:.4f}".format(kth, score))
    #    cv_scores.append(score)

    kf = KFold(X.shape[0], n_folds=n_folds, shuffle=True, random_state=777)
    for kth, (train_idx, test_idx) in enumerate(kf):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if skip_all_zero_target is not None and (y_train.sum() == 0 or y_test.sum() == 0):
            continue

        y_pred = task_class.solve(X_train, y_train, X_test)
        score = metrics(y_test, y_pred)
        l.info("KFold: ({0}) {1:.4f}".format(kth, score))
        cv_scores.append(score)
    mean_cv_score = np.mean(cv_scores)
    l.info("CV Score: {0:.4f} (var: {1:.6f})".format(
        mean_cv_score,
        np.var(cv_scores)))

    return mean_cv_score
