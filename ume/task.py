# -*- coding: utf-8 -*-
import json
import random
import logging as l
import os

import scipy.sparse as ss
import scipy.io as sio
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit as SSS
from sklearn.cross_validation import KFold
from sklearn.externals.joblib import Parallel, delayed

import ume
import ume.cross_validation
import ume.externals.jsonnet
from ume.utils import dynamic_load
from ume.metrics import multi_logloss


def hstack_mat(X, mat_fn, mat_name, conf=None):
    if mat_fn.endswith('.mat'):
        X_add = sio.loadmat(mat_fn)[mat_name]
        X_add = ss.csr_matrix(X_add)
    elif mat_fn.endswith('.npz'):
        X_add = np.load(mat_fn)[mat_name]
    else:
        raise RuntimeError("unsupported file")

    # slicing
    if conf is not None and 'slice' in conf:
        slice_start, slice_end = conf['slice']
        slice_start, slice_end = int(slice_start), int(slice_end)
        X_add = X_add[:, slice_start:slice_end]

    # horizontal stack
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
    """
    Make X from features in a model description.
    """
    X = None
    for mat_info in conf['features']:
        # dict format
        if isinstance(mat_info, dict):
            mat_fn = mat_info['file']
            mat_name = mat_info['name']
            X = hstack_mat(X, mat_fn, mat_name, conf=mat_info)
        # string format
        elif isinstance(mat_info, str) or isinstance(mat_info, unicode):
            X = hstack_mat(X, mat_info, 'X', conf=None)
        else:
            raise RuntimeError("Unsupported feature type: {0}".format(mat_info))

    if X is None:
        raise RuntimeError("Feature data is required")

    return X


def load_array(conf, name_path):
    """
    Load array from working data

    ```
    >> train_ids = load_array(self._conf, 'task.dataset.id_train')
    ```
    """
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


def _to_str_value(param_dict):
    # Primitive values
    if isinstance(param_dict, int):
        return param_dict
    elif isinstance(param_dict, str):
        return param_dict
    elif isinstance(param_dict, float):
        return param_dict

    converted_param_dict = {}
    for k, v in param_dict.items():
        if isinstance(v, int):
            converted_param_dict[k] = v
        elif isinstance(v, str):
            converted_param_dict[k] = v
        elif isinstance(v, float):
            converted_param_dict[k] = v
        elif isinstance(v, list):
            # convert recursively
            converted_param_dict[k] = [
                _to_str_value(elem)
                for elem in v
            ]
        elif isinstance(v, dict):
            # convert recursively
            converted_param_dict[k] = _to_str_value(v)
        else:
            # handle unicode for py27
            converted_param_dict[k] = str(v)
    return converted_param_dict


class TaskSpec(object):
    def __init__(self, jn):
        self._conf = self.__load_conf(jn)
        self._jn = jn

    def __load_conf(self, jn):
        json_dic = ume.externals.jsonnet.load(jn)
        if "ERROR" in json_dic:
            raise RuntimeError(json_dic)

        return json_dic

    def _load_model(self):
        model_klass = dynamic_load(self._conf['model']['class'])
        model_param = _to_str_value(self._conf['model'].get('params', {}))
        clf = model_klass(**model_param)
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

    def _to_output_fn(self, model_fn):
        output_fn = model_fn.replace(
            'data/input/model/',
            'data/output/')
        output_fn = output_fn + '.csv'
        return output_fn

    def create_submission(self, model_fn):
        """
        Called by `ume predict`
        task specified.
        """
        output_fn = self._to_output_fn(model_fn)

        self._create_submission(output_fn)
        self._post_processing(output_fn)

    def validate(self):
        """
        Called by `ume validation`
        """
        X_orig = make_X_from_features(self._conf)
        train_sz = len(load_array(self._conf, 'task.dataset.id_train'))
        X = X_orig[:train_sz, :]
        y = load_array(self._conf, 'task.dataset.y_train')
        y = y.reshape(y.size)

        cv_method_name = self._conf['task']['params']['validation']['class']
        cv_params_name = self._conf['task']['params']['validation'].get(
            'params', {})
        cv_params_name = _to_str_value(cv_params_name)

        cv_method = dynamic_load(cv_method_name)
        mean_cv_score = cv_method(X, y, self, **cv_params_name)

        task_metrics = self._conf['task']['params']['metrics']
        task_method = task_metrics['method']

        ume.db.add_validation_score(
            os.path.basename(self._jn),
            ume.__version__,
            task_method,
            mean_cv_score)


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
        l.info("Clf: {0}, X: {1}".format(str(clf), str(X_train.shape)))
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
        if isinstance(task_metrics, str):
            task_method = task_metrics
        elif isinstance(task_metrics, dict):
            task_method = task_metrics['method']
        else:
            raise RuntimeError("invalid task metrics")
        metrics = dynamic_load(task_method)

        cv_scores = []
        kf = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=777)
        for kth, (train_idx, test_idx) in enumerate(kf):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            y_pred = self.solve(X_train, y_train, X_test, y_test)

            score = metrics(y_test, y_pred)
            l.info("KFold: ({0}) {1:.4f}".format(kth, score))
            cv_scores.append(score)

        mean_cv_score = np.mean(cv_scores)
        l.info("CV Score: {0:.4f} (var: {1:.6f})".format(
            mean_cv_score,
            np.var(cv_scores)))

        ume.db.add_validation_score(
            os.path.basename(self._jn),
            ume.__version__,
            task_method,
            mean_cv_score)

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


class BinaryClassPredictProba(TaskSpec):
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
        l.info("Clf: {0}, X: {1}".format(str(clf), str(X_train.shape)))
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)
        #try:
        #    preds = clf.predict_proba(X_test)
        #except:
        #    preds = clf.decision_function(X_test)

        if len(preds.shape) > 1 and preds.shape[1] == 2:
            preds = preds[:, 1]

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
        df = pd.DataFrame(y_pred, columns=['Proba'])
        df['Id'] = test_ids.reshape(len(test_ids)).tolist()
        df.set_index('Id').to_csv(output_fn)


class Regression(TaskSpec):
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
        l.info("Clf: {0}, X: {1}".format(str(clf), str(X_train.shape)))
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
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
        df = pd.DataFrame(y_pred, columns=['Prediction'])
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
