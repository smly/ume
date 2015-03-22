# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from ume.externals._xgboost_wrapper import XGBoost as _XGBoost


class XGBoost(BaseEstimator):
    def __init__(self, **params):
        self.model = _XGBoost(**params)
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict_proba(self, X_test):
        pred = self.model.fit_predict(self.X, self.y, X_test)
        del self.model
        return pred
