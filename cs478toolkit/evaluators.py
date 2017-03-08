from __future__ import division
import numpy as np


def mse(predictions, targets):
    return ((np.atleast_1d(predictions) - np.atleast_1d(targets))**2).mean()


def rmse(predictions, targets):
    return np.sqrt(mse(predictions, targets))


def measure_error(predictions, targets, evaluator=mse):
    return evaluator(predictions, targets)


def measure_accuracy(predictions, targets, axis=1):
    equals_mask = np.equal(predictions, targets)
    equals_in_each = np.sum(equals_mask, axis=axis)
    accuracy = np.sum(equals_in_each == targets.shape[axis]) / targets.shape[0]
    return accuracy if accuracy < 1.0 else 1.0


def evaluate(data,
             targets,
             predict_function,
             measure_functions=None,
             *args):
    if not measure_functions:
        measure_functions = [measure_error, measure_accuracy]
    predictions = []
    for instance in data:
        predictions.append(predict_function(instance, *args))
    return (fn(predictions, targets) for fn in measure_functions)
