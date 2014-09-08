# coding: utf-8
import numpy as np


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
