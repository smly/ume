# -*- coding: utf-8 -*-
import numpy as np

from ume.externals.xgboost import XGBoost
from ume.metrics import multi_logloss


DERMATOLOGY_DATASET = "tests/dataset/dermatology.npz"


class TestMulticlassDataset(object):
    def test_multiclass_dataset(self):
        n = np.load(DERMATOLOGY_DATASET)
        clf = XGBoost(
            num_class=6,
            objective='multi:softprob',
            silent=1,
            seed=777,
            num_round=20)
        clf.fit(n['X_train'], n['y_train'])
        y_pred = clf.predict_proba(n['X_test'])

        score = multi_logloss(n['y_test'], y_pred)
        assert score < 0.3

    def test_xgboost_str(self):
        clf = XGBoost(num_round=100, seed=777)
        assert str(clf) == "XGBoost(num_round=100)"
