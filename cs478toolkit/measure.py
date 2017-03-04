import numpy as np
from __future__ import division


def mse(predictions, targets):
    return ((np.atleast_1d(predictions) - np.atleast_1d(targets))**2).mean()


def rmse(predictions, targets):
    return np.sqrt(mse(predictions, targets))


def error(predictions, targets, evaluator=mse):
    return evaluator(predictions, targets)


def accuracy(predictions, targets, axis=1):
    equals_mask = np.equal(predictions, targets)
    equals_in_each = np.sum(equals_mask, axis=axis)
    accuracy = np.sum(equals_in_each == targets.shape[axis]) / targets.shape[0]
    return accuracy if accuracy < 1.0 else 1.0
