# coding: utf-8
import numpy as np


def multi_logloss(y_true, y_pred, eps=1e-15):
    predictions = np.clip(y_pred, eps, 1 - eps)
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota


def rmse(ans, pred):
    """
    For https://www.kaggle.com/c/afsis-soil-properties
    """
    return np.sqrt(np.mean((ans - pred) ** 2))


def apk_score(ans, pred, k=32500):
    """
    For https://www.kaggle.com/c/avito-prohibited-content
    """
    assert len(ans) == len(pred)
    pred_idxlist = list(map(
        lambda x: x[0],
        sorted(enumerate(pred), key=lambda x: x[1], reverse=True)))
    ans_idxlist = []
    for i, res in enumerate(ans.tolist()):
        if res == 1:
            ans_idxlist.append(i)
    return _apatk(pred_idxlist, ans_idxlist, k)


def _apatk(predictions, solution, K):
    countRelevants = 0
    listOfPrecisions = list()
    for i, line in enumerate(predictions):
        currentk = i + 1.0
        if line in solution:
            countRelevants += 1
        precisionAtK = countRelevants / currentk
        listOfPrecisions.append(precisionAtK)
        if currentk == K:
            break

    return sum(listOfPrecisions) / K
