from __future__ import division, print_function
import argparse
import arff
import numpy as np
import re


def percent(value):
    value = float(value)
    if not 0 < value <= 100:
        raise argparse.ArgumentError()
    if value > 1:
        value /= 100.0
    return value


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Toolkit for BYU CS 478",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help='increase verbosity')
    parser.add_argument(
        '-f',
        '--file',
        metavar='FILE',
        required=True,
        help='path to ARFF file to load')
    parser.add_argument(
        '-l',
        '--layers',
        metavar='LAYER',
        nargs='+',
        type=int,
        required=True,
        help='layer sizes: <input> <hidden>... <output>')
    parser.add_argument(
        '-c',
        '--checkpoint',
        metavar='FILE',
        help='checkpoint file to save/load weights, biases etc. from')
    parser.add_argument(
        '-t',
        '--training',
        type=percent,
        default=.75,
        metavar='PERCENT',
        help='percentage of entire dataset for the training set size')
    parser.add_argument(
        '-r', '--learning-rate', type=float, default=.1, help='learning rate')
    parser.add_argument('-s', '--seed', help='random seed')
    return parser


def parse_args(parser=_parse_args):
    args, _ = parser().parse_known_args()
    if args.seed:
        # Use a seed for deterministic results
        np.random.seed(int(args.seed))
    return args


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


def shuffle(features, labels):
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "Features {} and labels {} must have the same number of rows".
            format(features.shape, labels.shape))
    permutation = np.random.permutation(features.shape[0])
    return features[permutation], labels[permutation]


def _split(data, label_size):
    return data[:, :-label_size], data[:, -label_size:]


def split(features, labels, percent):
    percent = float(percent)
    if not 0 < percent < 1:
        raise ValueError("percent must be in range: 0-1")
    index = int(percent * features.shape[0])
    return features[:index], features[index:], labels[:index], labels[index:]


# HACK: arff.load only accepts an open file descriptor and BYU CS uses a custom arff format
def _fix_attribute_types(f):
    # TODO: do not load entire contents of file into RAM at once
    f.seek(0)
    s = f.read()
    f.seek(0)
    s = re.sub(r'continuous', 'numeric', s, flags=re.IGNORECASE)
    f.write(s)
    f.truncate()
    f.seek(0)


def load(file_path, label_size=1, encode_nominal=True, add_bias=False):
    with open(file_path, 'r+') as f:
        try:
            arff_data = arff.load(f, encode_nominal)
        except arff.BadAttributeType:
            _fix_attribute_types(f)
            arff_data = arff.load(f, encode_nominal)

    data = np.array(arff_data['data'])
    np.random.shuffle(data)

    if add_bias:
        data = np.insert(data, -label_size, 1, axis=1)

    return _split(data, label_size)
