# -*- coding: utf-8 -*-
from sklearn.ensemble import BaggingRegressor as SK_BaggingRegressor
from sklearn.base import BaseEstimator
from ume.utils import dynamic_load


class BaggingRegressor(BaseEstimator):
    """
    Sample:

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
