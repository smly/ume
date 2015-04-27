# -*- coding: utf-8 -*-
from sklearn.ensemble import BaggingRegressor as SK_BaggingRegressor
from sklearn.base import BaseEstimator
from ume.utils import dynamic_load

import numpy as np


class BaggingRegressor(BaseEstimator):
    """
    Usage:

    ```
    "model": {
        "class": "ume.ensemble.BaggingRegressor",
        "params": {
            "base_estimator": {
                "class": "sklearn.svm.SVR",
                "params": {
                    "kernel": "rbf",
                    "degree": 1,
                    "C": 1000000.0,
                    "epsilon": 0.01,
                },
            },
            "bag_kwargs": {
                "n_estimators": 100,
                "n_jobs": 5,
                "max_samples": 0.9,
            },
        }
    }
    ```
    """
    def __init__(self, base_estimator=None, bag_kwargs=None):
        klass = dynamic_load(base_estimator['class'])
        svr_reg = klass(**base_estimator['params'])
        self.__clf = SK_BaggingRegressor(base_estimator=svr_reg, **bag_kwargs)

    def fit(self, X, y):
        return self.__clf.fit(X, y)

    def predict(self, X):
        return self.__clf.predict(X)


class WeightedAverage(BaseEstimator):
    """
    Usage:

    ```
    "model": {
        "class": "ume.ensemble.WeightedAverage",
        "params": {
            "weight_list": [
                0.5,
                0.5
            ],
            "estimator_list": [
                {
                    "class": "sklearn.svm.SVR",
                    "params": {
                        "kernel": "rbf",
                        "degree": 1,
                        "C": 1000000.0,
                        "epsilon": 0.01,
                    },
                },
                {
                    "class": "sklearn.svm.SVR",
                    "params": {
                        "kernel": "linear",
                        "degree": 1,
                        "C": 1000000.0,
                        "epsilon": 0.01,
                    },
                },
            ],
        }
    }
    ```
    """
    def __init__(self, estimator_list, weight_list=None):
        self.__clf_list = []
        for estimator_class in estimator_list:
            klass = dynamic_load(estimator_class['class'])
            clf = klass(**estimator_class['params'])
            self.__clf_list.append(clf)

        self.__weight = weight_list

    def fit(self, X, y):
        for clf in self.__clf_list:
            clf.fit(X, y)

    def predict(self, X):
        ans = [
            self.__weight[i] * clf.predict(X)
            for i, clf in enumerate(self.__clf_list)
        ]

        return np.sum(ans, axis=0)
