# -*- ocding: utf-8 -*-
import random
import logging as l

import numpy as np
import scipy.sparse as ss
import pandas as pd
import theano
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.base import BaseEstimator
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from lasagne.updates import adagrad
from nolearn.lasagne import NeuralNet
from ume.metrics import multi_logloss
from ume.externals.xgboost import XGBoost as XGB
from ume.utils import dynamic_load


class AdjustVariable(object):
    """
    * ref: http://danielnouri.org/notes/category/deep-learning/
    """
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class TwoLayerNeuralNetwork(BaseEstimator):
    def __init__(self, **params):
        self.seed = params.get('seed', 777)

        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Prepare network parameters
        network_params = {}

        # Input shape
        network_params['input_shape'] = (None, params['num_features'])

        # Prepare layers
        network_params['layers'] = []
        for param_layer in params['layers']:
            network_params['layers'].append((
                param_layer['name'],
                dynamic_load(param_layer['class'])
            ))
            for k in param_layer.keys():
                if k in ['name', 'class']:
                    continue
                name = "{0}_{1}".format(param_layer['name'], k)
                network_params[name] = param_layer[k]

        # Prepare outputs
        network_params['output_nonlinearity'] = (
            dynamic_load(params['output_nonlinearity']))

        # Prepare updates
        if 'update' in params:
            network_params['update'] = dynamic_load(params['update'])
        if 'update_learning_rate' in params:
            update_learning_rate = params['update_learning_rate']
            network_params['update_learning_rate'] = (
                theano.shared(np.cast['float32'](update_learning_rate))
            )
        if 'adjust_variable' in params:
            if params['adjust_variable'] == 1:
                adjuster_start = params.get('adjuster_start', 0.01)
                adjuster_stop = params.get('adjuster_stop', 0.001)

                network_params['on_epoch_finished'] = [
                    AdjustVariable('update_learning_rate',
                        start=adjuster_start,
                        stop=adjuster_stop)
                ]

        # Other parameters
        network_params['eval_size'] = params.get('eval_size', 0.01)
        network_params['verbose'] = params.get('verbose', 1)
        network_params['max_epochs'] = params.get('max_epochs', 100)

        # Create a clf object
        self.clf = NeuralNet(**network_params)

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict_proba(self, X_test):
        return self.clf.predict_proba(X_test)
