# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from ume.externals._xgboost_wrapper import XGBoost as _XGBoost


class XGBoost(BaseEstimator):
    def __init__(self, **params):
        self.model = _XGBoost(**params)
        self.params = params
        self.X = None
        self.y = None

    def __str__(self):
        model_parameters = [
            'objective', 'max_depth', 'min_child_weight',
            'num_round', 'subsample']
        param_str = ", ".join(
            ["{0}={1}".format(k, self.params[k])
            for k in sorted(self.params.keys())
            if k in model_parameters])
        return "XGBoost({0})".format(param_str)

    def fit(self, X, y):
        self.X = X
        self.y = y

    def _set_test_label(self, y_test):
        self.model.set_test_label(y_test)

    def predict_proba(self, X_test):
        pred = self.model.fit_predict(self.X, self.y, X_test)
        del self.model
        return pred
