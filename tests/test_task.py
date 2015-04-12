# -*- coding: utf-8 -*-
import tempfile
import os

import numpy as np

import ume.task


def test_hstack_mat_direct_path():
    X = None

    with tempfile.TemporaryDirectory() as temp_dirname:
        mat_path = os.path.join(temp_dirname, 'testcase.npz')

        # generate working file
        X_sample = np.array([[1, 2, 3], [4, 5, 6]])
        np.savez(mat_path, X=X_sample)

        # load working file
        X = ume.task.hstack_mat(X, mat_path, 'X')
        assert isinstance(X, np.ndarray)
        assert X.shape == (2, 3)


def test_hstack_mat_slice():
    X = None

    with tempfile.TemporaryDirectory() as temp_dirname:
        mat_path = os.path.join(temp_dirname, 'testcase.npz')
        conf = {"slice": [1, 3]}

        # generate working file
        X_sample = np.array([[1, 2, 3], [4, 5, 6]])
        np.savez(mat_path, X=X_sample)

        # load working file
        X = ume.task.hstack_mat(X, mat_path, 'X', conf=conf)
        assert isinstance(X, np.ndarray)
        assert X.shape == (2, 2)
        assert X[0, 0] == 2
