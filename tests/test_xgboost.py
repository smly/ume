# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as ss

from ume.externals._xgboost_wrapper import XGBoost


class TestXGBoost(object):
    def test_nd_load(self):
        clf = XGBoost(silent=1)
        X_train = np.array([
            [1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1 ,1, 1],
        ])
        y = np.array([1.0, 1.0, 0.0, 0.0])
        pred = clf.fit_predict(X_train, y, X_train)

        row_sum = pred.sum(axis=1)
        assert np.sum(row_sum - np.ones(4)) < 1e-5

        assert pred.shape[0] == 4
        assert pred.shape[1] == 1
        assert pred[0] > 0.5
        assert pred[1] > 0.5
        assert pred[2] < 0.5
        assert pred[3] < 0.5

    def test_ss_load(self):
        clf = XGBoost(silent=1)
        X_train = ss.csr_matrix(np.array([
            [1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1 ,1, 1],
        ]))
        y = np.array([1.0, 1.0, 0.0, 0.0])
        pred = clf.fit_predict(X_train, y, X_train)

        row_sum = pred.sum(axis=1)
        assert np.sum(row_sum - np.ones(4)) < 1e-5

        assert pred.shape[0] == 4
        assert pred.shape[1] == 1
        assert pred[0] > 0.5
        assert pred[1] > 0.5
        assert pred[2] < 0.5
        assert pred[3] < 0.5
