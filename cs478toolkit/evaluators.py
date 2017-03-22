from __future__ import division, print_function
import numpy as np


def _ensure_same_shape(predictions, targets):
    predictions = np.atleast_1d(predictions).flatten()
    targets = np.atleast_1d(targets).flatten()

    if predictions.shape != targets.shape:
        msg = "shape mismatch: cannot compare objects with different shapes"
        raise ValueError(msg)

    return predictions, targets


def mse(predictions, targets):
    predictions, targets = _ensure_same_shape(predictions, targets)
    return ((predictions - targets)**2).mean()


def rmse(predictions, targets):
    return np.sqrt(mse(predictions, targets))


def measure_error(predictions, targets, evaluator=mse):
    return evaluator(predictions, targets)


def measure_accuracy(predictions, targets):
    predictions, targets = _ensure_same_shape(predictions, targets)
    accuracy = np.count_nonzero(predictions == targets) / len(targets)
    return np.minimum(1.0, accuracy)


def evaluate(data,
             targets,
             predict_function,
             measure_functions=None,
             progress=False,
             *args):
    if not measure_functions:
        measure_functions = [measure_error, measure_accuracy]

    it = data
    if progress:
        try:
            import tqdm
            it = tqdm.tqdm(data)
        except ImportError:
            print("install tqdm to report progress")

    predictions = []
    for instance in it:
        predictions.append(predict_function(instance, *args))

    return (fn(predictions, targets) for fn in measure_functions)
