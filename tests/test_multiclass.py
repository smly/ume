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
            num_round=20)
        clf.fit(n['X_train'], n['y_train'])
        y_pred = clf.predict_proba(n['X_test'])

        score = multi_logloss(n['y_test'], y_pred)
        print(score)
        assert score < 0.3
