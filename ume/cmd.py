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
import ume.db
from ume.utils import (
    feature_functions,
    save_npz,
    save_mat,
    load_mat,
    dynamic_load,
    load_settings
)
from ume.task import TaskSpec


def parse_args():
    p = argparse.ArgumentParser(
        description='CLI interface of Ume')
    p.add_argument(
        '--version',
        dest='version',
        action='store_true',
        default=False
    )

    subparsers = p.add_subparsers(
        dest='subparser_name',
        help='sub-commands for instant action')

    # feature
    f_parser = subparsers.add_parser('feature')
    f_parser.add_argument('-a', '--all', action='store_true', default=False)
    f_parser.add_argument('-n', '--name', type=str, required=True)

    # initialize
    subparsers.add_parser('init')

    # validate
    v_parser = subparsers.add_parser('validate')
    v_parser.add_argument(
        '-m', '--model',
        required=True,
        type=str,
        help='model description file described by json format')

    # predict
    p_parser = subparsers.add_parser('predict')
    p_parser.add_argument(
        '-m', '--model',
        required=True,
        type=str,
        help='model description file described by json format')

    return p


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
        "data/output",
        "data/working",
        "note",
        "trunk",
    ]
    for path in dirs:
        _makedirs(path)


def run_validation(args):
    conf = load_settings(args.model)
    klass = dynamic_load(conf['task']['class'])
    task = klass(args.model)
    task.validate()


def run_prediction(args):
    conf = load_settings(args.model)
    klass = dynamic_load(conf['task']['class'])
    task = klass(args.model)
    task.create_submission(args.model)


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
    p = parse_args()
    args = p.parse_args()

    if args.subparser_name == 'init':
        if ume.db.exists_sqlitedb():
            l.warning("umedb already exists.")
            sys.exit(1)
        else:
            run_initialize(args)
            ume.db.init_db()
            sys.exit(0)

    if not ume.db.exists_sqlitedb():
        l.warning("umedb doesn't exists. Create new db now.")
        ume.db.init_db()

    if args.version:
        run_version_checker(args)
    elif args.subparser_name == 'validate':
        run_validation(args)
    elif args.subparser_name == 'predict':
        run_prediction(args)
    elif args.subparser_name == 'feature':
        run_feature(args)
    else:
        p.print_help()
