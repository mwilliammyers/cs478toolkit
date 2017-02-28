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
    parser = argparse.ArgumentParser(description="Toolkit for BYU CS 478")
    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help='Increase verbosity')
    # parser.add_argument('-N', '--normalize', action='store_true', help='Use normalized data')
    parser.add_argument('-R', '--seed', help='Random seed')
    parser.add_argument(
        '-f',
        '--file',
        metavar='FILE',
        required=True,
        help='Path to ARFF file to load')
    parser.add_argument(
        '-l',
        '--layers',
        metavar='LAYER',
        nargs='+',
        type=int,
        required=True,
        help='Layer sizes, in the format: <input> <hidden>... <output>')
    parser.add_argument(
        '-c',
        '--checkpoint',
        metavar='FILE',
        help='Checkpoint file to load weights, biases etc. from')
    parser.add_argument(
        '-s',
        '--training',
        type=percent,
        metavar='PERCENT',
        help='Percentage of entire dataset for the training set size')
    # parser.add_argument('-E', metavar=('METHOD', 'args'), required=True, nargs='+', help="Evaluation method (training | static <test_ARFF_file> | random <%%_for_training> | cross <num_folds>)")
    return parser


def parse_args(parser=_parse_args):
    args, _ = parser().parse_known_args()
    if args.seed:
        # Use a seed for deterministic results
        # random.seed(args.seed)
        np.random.seed(int(args.seed))
    return args


def mse(predictions, targets):
    return ((np.atleast_1d(predictions) - np.atleast_1d(targets))**2).mean()


def rmse(predictions, targets):
    return np.sqrt(mse(predictions, targets))


def measure_accuracy(predictions, targets, evaluator=mse):
    return evaluator(predictions, targets)


def shuffle(features, labels):
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "Features {} and labels {} must have the same number of rows".
            format(features.shape, labels.shape))
    permutation = np.random.permutation(features.shape[0])
    return features[permutation], labels[permutation]


def _split(data, label_size):
    return data[:, :-label_size], data[:, -label_size:]


def split(features, labels, training_percent):
    training_percent = float(training_percent)
    if not 0 < training_percent < 1:
        raise ValueError("training_percent must be in range: 0-1")
    index = int(training_percent * features.shape[0])
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

    if add_bias:
        data = np.insert(data, -label_size, 1, axis=1)

    return _split(data, label_size)
