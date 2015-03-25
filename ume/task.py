# -*- coding: utf-8 -*-
import json
import random
import logging as l

import scipy.sparse as ss
import scipy.io as sio
import pandas as pd
import numpy as np
import jsonnet
from sklearn.cross_validation import KFold
from sklearn.externals.joblib import Parallel, delayed
from ume.utils import dynamic_load
from ume.metrics import multi_logloss


def hstack_mat(X, mat_fn, mat_name):
    if mat_fn.endswith('.mat'):
        X_add = sio.loadmat(mat_fn)[mat_name]
        X_add = ss.csr_matrix(X_add)
    elif mat_fn.endswith('.npz'):
        X_add = np.load(mat_fn)[mat_name]
    else:
        raise RuntimeError("unsupported file")

    if X is None:
        return X_add
    else:
        if isinstance(X, np.ndarray) and isinstance(X_add, np.ndarray):
            return np.hstack([X, X_add])
        elif isinstance(X, ss.csr_matrix) or isinstance(X, ss.csc_matrix):
            return ss.csr_matrix(
                ss.hstack([X, ss.csr_matrix(X_add)])
            )
        else:
            raise RuntimeError("Unsupported datatype")


def make_X_from_features(conf):
    X = None
    for mat_info in conf['features']:
        if isinstance(mat_info, dict):
            mat_fn = mat_info['file']
            mat_name = mat_info['name']
            X = hstack_mat(X, mat_fn, mat_name)
        elif isinstance(mat_info, str):
            X = hstack_mat(X, mat_info, 'X')
        else:
            raise RuntimeError("Unsupported feature type: {0}".format(mat_info))

    if X is None:
        raise RuntimeError("Feature data is required")

    return X


def load_array(conf, name_path):
    arr_ = dict(conf)
    for name in name_path.split('.'):
        arr_ = arr_[name]

    if isinstance(arr_, dict):
        mat_fn = arr_['file']
        mat_name = arr_['name']

        return hstack_mat(None, mat_fn, mat_name)
    elif isinstance(arr_, str):
        return hstack_mat(None, arr_, 'X')
    else:
        raise RuntimeError("Unsupported feature type: {0}".format(mat_info))


class TaskSpec(object):
    def __init__(self, jn):
        self._conf = self.__load_conf(jn)

    def __load_conf(self, jn):
        json_str = jsonnet.load(jn).decode('utf-8')
        if "ERROR" in json_str:
            raise RuntimeError(json_str)
        try:
            json_dic = json.loads(json_str)
        except:
            raise RuntimeError("Broken json data")

        return json_dic

    def _load_model(self):
        model_klass = dynamic_load(self._conf['model']['class'])
        clf = model_klass(**(self._conf['model'].get('params', {})))
        l.info("Clf: {0}".format(str(clf)))
        return clf

    def solve(self, X_train, y_train, X_test):
        raise NotImplementedError("Need to implement `solve`.")

    def _create_submission(self, output_fn):
        raise NotImplementedError("Need to implement `_create_submission`.")

    def _post_processing(self, output_fn):
        if not 'task' in self._conf: return
        if not 'params' in self._conf['task']: return
        if not 'postprocessing' in self._conf['task']['params']: return
        method = dynamic_load(self._conf['task']['params']['postprocessing'])
        method(output_fn)

    def create_submission(self, output_fn):
        """
        called by `ume predict`
        task specified.
        """
        self._create_submission(output_fn)
        self._post_processing(output_fn)

    def validate(self):
        """
        called by `ume validation`
        """
        X_orig = make_X_from_features(self._conf)
        train_sz = len(load_array(self._conf, 'task.dataset.id_train'))
        X = X_orig[np.array(range(train_sz)), :]
        y = load_array(self._conf, 'task.dataset.y_train')
        y = y.reshape(y.size)

        task_metrics = self._conf['task']['params']['metrics']
        metrics = dynamic_load(task_metrics.get('method', task_metrics))

        cv_scores = []
        kf = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=777)
        for kth, (train_idx, test_idx) in enumerate(kf):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            y_pred = self.solve(X_train, y_train, X_test)
            score = metrics(y_test, y_pred)
            l.info("KFold: ({0}) {1:.4f}".format(kth, score))
            cv_scores.append(score)

        l.info("CV Score: {0:.4f} (var: {1:.6f})".format(
            np.mean(cv_scores),
            np.var(cv_scores)))


class MultiClassPredictProba(TaskSpec):
    def __init__(self, jn):
        self.required = ['features', 'model', 'task']

        # Load jsonnet config
        TaskSpec.__init__(self, jn)

        # Check fields
        for field in self.required:
            if field not in self._conf.keys():
                raise RuntimeError("Required field: {0}".format(field))

    def solve(self, X_train, y_train, X_test):
        clf = self._load_model()
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)
        del clf
        return preds

    def _create_submission(self, output_fn):
        X_orig = make_X_from_features(self._conf)
        train_ids = load_array(self._conf, 'task.dataset.id_train')
        test_ids = load_array(self._conf, 'task.dataset.id_test')
        train_sz = len(train_ids)
        test_sz = len(test_ids)

        X_train = X_orig[np.array(range(train_sz)), :]
        X_test = X_orig[np.array(range(train_sz, train_sz + test_sz)), :]
        y = load_array(self._conf, 'task.dataset.y_train')
        y = y.reshape(y.size)

        y_pred = self.solve(X_train, y, X_test)
        df = pd.DataFrame(y_pred, columns=[
            'Class_{0}'.format(i + 1)
            for i in range(y_pred.shape[1])])
        df['Id'] = test_ids.reshape(len(test_ids)).tolist()
        df.set_index('Id').to_csv(output_fn)


class DebugMultiClassPredictProba(TaskSpec):
    def __init__(self, jn):
        self.required = ['features', 'model', 'task']

        # Load jsonnet config
        TaskSpec.__init__(self, jn)

        # Check fields
        for field in self.required:
            if field not in self._conf.keys():
                raise RuntimeError("Required field: {0}".format(field))

    def solve(self, X_train, y_train, X_test, y_test):
        clf = self._load_model()
        clf.fit(X_train, y_train)
        clf._set_test_label(y_test)
        preds = clf.predict_proba(X_test)
        del clf
        return preds

    def validate(self):
        X_orig = make_X_from_features(self._conf)
        train_sz = len(load_array(self._conf, 'task.dataset.id_train'))
        X = X_orig[np.array(range(train_sz)), :]
        y = load_array(self._conf, 'task.dataset.y_train')
        y = y.reshape(y.size)

        task_metrics = self._conf['task']['params']['metrics']
        metrics = dynamic_load(task_metrics.get('method', task_metrics))

        cv_scores = []
        kf = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=777)
        for kth, (train_idx, test_idx) in enumerate(kf):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            y_pred = self.solve(X_train, y_train, X_test, y_test)

            score = metrics(y_test, y_pred)
            l.info("KFold: ({0}) {1:.4f}".format(kth, score))
            cv_scores.append(score)

        l.info("CV Score: {0:.4f} (var: {1:.6f})".format(
            np.mean(cv_scores),
            np.var(cv_scores)))

    def _create_submission(self, output_fn):
        X_orig = make_X_from_features(self._conf)
        train_ids = load_array(self._conf, 'task.dataset.id_train')
        test_ids = load_array(self._conf, 'task.dataset.id_test')
        train_sz = len(train_ids)
        test_sz = len(test_ids)

        X_train = X_orig[np.array(range(train_sz)), :]
        X_test = X_orig[np.array(range(train_sz, train_sz + test_sz)), :]
        y = load_array(self._conf, 'task.dataset.y_train')
        y = y.reshape(y.size)

        y_pred = self.solve(X_train, y, X_test)
        df = pd.DataFrame(y_pred, columns=[
            'Class_{0}'.format(i + 1)
            for i in range(y_pred.shape[1])])
        df['Id'] = test_ids.reshape(len(test_ids)).tolist()
        df.set_index('Id').to_csv(output_fn)


def main():
    task = MultiClassPredictProba("data/input/model/xgb.jn")
    task.validate()
    task.create_submission()


if __name__ == '__main__':
    l.basicConfig(format=u'[%(asctime)s] %(message)s', level=l.INFO)
    random.seed(777)
    main()
