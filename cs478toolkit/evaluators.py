from __future__ import division
import numpy as np


def mse(predictions, targets):
    return ((np.atleast_1d(predictions) - np.atleast_1d(targets))**2).mean()


def rmse(predictions, targets):
    return np.sqrt(mse(predictions, targets))


def measure_error(predictions, targets, evaluator=mse):
    # print(predictions, targets)
    return evaluator(predictions, targets)


def measure_accuracy(predictions, targets):
    accuracy = np.count_nonzero(predictions == targets) / len(targets)
    return np.minimum(1.0, accuracy)


def evaluate(data, targets, predict_function, measure_functions=None, *args):
    if not measure_functions:
        measure_functions = [measure_error, measure_accuracy]
    predictions = []
    for instance, target in zip(data, targets):
        p = predict_function(instance, *args)
        # print(instance, p, target)
        predictions.append(p)
    return (fn(predictions, targets) for fn in measure_functions)
