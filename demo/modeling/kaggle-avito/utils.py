# -*- coding: utf-8 -*-
import logging as l

from ume.utils import dynamic_load


class PredictProba(object):
    def __init__(self, settings):
        self.metrics = settings['metrics']
        self.prediction = settings['prediction']
        self.model = settings['model']

    def solve(self, X_train, X_test, y_train):
        klass = dynamic_load(self.model['class'])
        clf = klass(**self.model['params'])
        l.info("Training model: {0}".format(str(clf)))
        clf.fit(X_train, y_train)
        l.info("Training model ... done")
        y_pred = clf.predict_proba(X_test)[:, 1]

        return y_pred
