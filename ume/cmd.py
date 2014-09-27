# -*- coding: utf-8 -*-
import logging as l
import argparse
import os
import sys
import json

import numpy as np
import scipy as sp
import scipy.sparse as ss
import scipy.io as sio
import pandas as pd
import jsonnet
import sklearn

import ume
from ume.utils import (
    feature_functions,
    save_npz,
    save_mat,
    load_mat,
    dynamic_load,
    load_settings
)
from ume.visualize import Plot


def parse_args():
    p = argparse.ArgumentParser(
        description='CLI interface UME')
    p.add_argument(
        '--version',
        dest='version',
        action='store_true',
        default=False
    )

    subparsers = p.add_subparsers(
        dest='subparser_name',
        help='sub-commands for instant action')

    f_parser = subparsers.add_parser('feature')
    f_parser.add_argument('-a', '--all', action='store_true', default=False)
    f_parser.add_argument('-n', '--name', type=str, required=True)

    subparsers.add_parser('init')

    v_parser = subparsers.add_parser('validate')
    v_parser.add_argument(
        '-m', '--model',
        required=True,
        type=str,
        help='model description file described by json format')

    z_parser = subparsers.add_parser('visualize')
    z_parser.add_argument('-j', '--json', type=str, required=True)
    z_parser.add_argument('-o', '--output', type=str, required=True)

    p_parser = subparsers.add_parser('predict')
    p_parser.add_argument(
        '-m', '--model',
        required=True,
        type=str,
        help='model description file described by json format')
    p_parser.add_argument(
        '-o', '--output',
        required=True,
        type=str,
        help='output file')

    return p.parse_args()


def run_visualization(args):
    if args.json.endswith(".jsonnet"):
        config = json.loads(jsonnet.load(args.json).decode())
    else:
        with open(args.json, 'r') as f:
            config = json.load(f)

    title_name = config['title'] if 'title' in config else ""
    p = Plot(title=title_name)
    data_dict = {}
    for source_name in config['datasource'].keys():
        data_dict[source_name] = pd.read_csv(config['datasource'][source_name])

    for i, plotdata in enumerate(config['plotdata']):
        if 'plot' not in plotdata:
            continue  # empty space

        for j, plate in enumerate(plotdata['plot']):
            plate_source = data_dict[plate['source']]
            for ax_name in ['X', 'y']:
                if ax_name == 'y' and ax_name not in plate:
                    # plate_hist doesn't require y-axis.
                    config['plotdata'][i]['plot'][j][ax_name] = None
                else:
                    col = plate[ax_name]
                    config['plotdata'][i]['plot'][j][ax_name] = plate_source[col]

    layout_param = {} if 'layout' not in config else config['layout']

    for c in config['plotdata']:
        p.add(c)
    p.save(args.output, **layout_param)


def _save_mat_or_npz(target, result):
    result_keys = result.keys()
    if len(result) == 0:
        raise RuntimeError("No featur selected")

    first_elem = list(result_keys)[0]
    if type(result[first_elem]) is np.ndarray:
        save_npz(target, result)
    else:
        save_mat(target, result)


def run_feature(args):
    if args.all is True:
        print(args.name)
        mod, names = feature_functions(args.name)
        for name in names:
            target = "{0}.{1}".format(args.name, name)
            l.info("Feature generation: {0}".format(target))
            func = getattr(mod, name)
            result = func()
            _save_mat_or_npz(target, result)
    else:
        l.info("Feature generation: {0}".format(args.name))
        klass = dynamic_load(args.name)
        result = klass()
        _save_mat_or_npz(args.name, result)


def _makedirs(relative_path):
    pwd = os.getcwd()
    full_path = os.path.join(pwd, relative_path)
    if not os.path.exists(full_path):
        os.makedirs(os.path.join(pwd, relative_path))


def run_initialize(args):
    dirs = [
        "data/input/model",
        "data/input/visualize",
        "data/output",
        "data/output/visualize",
        "data/working",
        "note",
        "trunk",
    ]
    for path in dirs:
        _makedirs(path)


def _load_features(f_names):
    X = None
    for f_name in f_names:
        l.info(f_name)
        var_name = 'X'
        if type(f_name) is dict:
            var_name = f_name['name']
            f_name = f_name['file']

        X_add = load_mat(f_name)[var_name]
        if X is None:
            X = X_add
        elif type(X) is np.ndarray and type(X_add) is np.ndarray:
            X = np.hstack((X, X_add))
        else:
            X = X_add if X is None else ss.hstack((X, X_add))
    return X


def _load_train_test(settings):
    X = _load_features(settings['features'])

    idx_train = load_mat(settings['idx']['train']['file'])[
        settings['idx']['train']['name']
    ]
    idx_test = load_mat(settings['idx']['test']['file'])[
        settings['idx']['test']['name']
    ]
    idx_train = idx_train[:, 0]
    idx_test = idx_test[:, 0]
    X_train = X[idx_train]
    X_test = X[idx_test]
    y_train = load_mat(settings['target']['file'])[
        settings['target']['name']
    ]
    #y_train = y_train[:, 0, 0]
    y_train = y_train[:, 0]
    return X_train, X_test, y_train


def run_validation(args):
    settings = load_settings(args.model)
    l.info("Loading dataset")
    X, _, y = _load_train_test(settings)
    kfoldcv = dynamic_load(settings['cross_validation']['method'])
    score, variance = kfoldcv(X, y, settings)
    l.info("CV score: {0:.4f} (var: {1:.6f})".format(score, variance))


def run_prediction(args):
    settings = load_settings(args.model)
    prediction = settings['prediction']
    l.info("Loading dataset")
    X_train, X_test, y_train = _load_train_test(settings)

    predict_klass = dynamic_load(prediction['method'])
    p = predict_klass(settings)
    y_pred = p.solve(X_train, X_test, y_train)

    pd.DataFrame({'y_pred': y_pred}).to_csv(args.output, index=False)


def run_version_checker(args):
    version_fmt = "{name:s} v{ver:s}"
    print(version_fmt.format(name='ume', ver=ume.__version__))
    print(version_fmt.format(name='numpy', ver=np.__version__))
    print(version_fmt.format(name='scipy', ver=sp.__version__))
    print(version_fmt.format(name='pandas', ver=pd.__version__))
    print(version_fmt.format(name='scikit-learn', ver=sklearn.__version__))
    print(version_fmt.format(name='jsonnet', ver=jsonnet.__version__))


def main():
    l.basicConfig(format='%(asctime)s %(message)s', level=l.INFO)
    sys.path.append(os.getcwd())
    args = parse_args()
    if args.version:
        run_version_checker(args)
    elif args.subparser_name == 'validate':
        run_validation(args)
    elif args.subparser_name == 'visualize':
        run_visualization(args)
    elif args.subparser_name == 'predict':
        run_prediction(args)
    elif args.subparser_name == 'feature':
        run_feature(args)
    elif args.subparser_name == 'init':
        run_initialize(args)
    else:
        raise RuntimeError("No such sub-command.")
