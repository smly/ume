# -*- coding: utf-8 -*-
import sklearn.datasets
import numpy as np
from sklearn.decomposition import PCA


def gen_first_two():
    np.random.seed(seed=777)
    iris = sklearn.datasets.load_iris()

    X = iris.data[:, :2]
    y = iris.target
    y[np.where(y == 0)] = 0
    y[np.where(y != 0)] = 1
    id_list = np.array(["Line{}".format(i) for i in range(len(y))])

    idx = np.arange(len(y))
    np.random.shuffle(idx)

    idx_train = idx[:100]
    idx_test = idx[100:]
    y_train = y[idx_train]

    # id name (used for submission file)
    id_train = id_list[idx_train]
    id_test = id_list[idx_test]

    return {
        'X': X,  # np.vstack([X_train, X_test])
        'y': y_train,  # for training data only
        'id_train': id_train,
        'id_test': id_test,

        # not used in ume
        'idx_train': idx_train,
        'idx_test': idx_test,
    }


def gen_pca():
    #npz = np.load('data/working/feat.gen_first_two.npz')
    npz = gen_first_two()
    idx_train = npz['idx_train']
    idx_test = npz['idx_test']

    iris = sklearn.datasets.load_iris()
    X = iris.data
    X = np.vstack([X[idx_train], X[idx_test]])  # re-ordering

    X_reduced = PCA(n_components=3).fit_transform(X)
    return {'X': X_reduced}
