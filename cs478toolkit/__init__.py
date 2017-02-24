import argparse
import arff
import numpy as np
import logging


def _log_level(count):
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    return levels[min(len(levels) - 1, count)]


def parse_args():
    parser = argparse.ArgumentParser(description="Toolkit for BYU CS 478")

    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help='Increase verbosity')
    # parser.add_argument('-N', '--normalize', action='store_true', help='Use normalized data')
    parser.add_argument('-R', '--seed', help='Random seed')
    # parser.add_argument('-L', required=True, choices=['baseline', 'perceptron', 'neuralnet', 'decisiontree', 'knn'], help='Learning Algorithm')
    parser.add_argument(
        '-A',
        '--arff',
        metavar='filename',
        required=True,
        help='Path to ARFF file to load')
    # parser.add_argument('-E', metavar=('METHOD', 'args'), required=True, nargs='+', help="Evaluation method (training | static <test_ARFF_file> | random <%%_for_training> | cross <num_folds>)")

    return parser.parse_args()


def load_data(file, add_bias=True, encode_nominal=True):
    arff_data = arff.load(open(file, 'r'), encode_nominal)
    data = np.array(arff_data['data'])
    if add_bias:
        data = np.insert(data, -1, 1, axis=1)
    return data


def initialize(
        log_format="%(filename)s:%(lineno)s:%(funcName)s():\n%(message)s",
        args_parser=parse_args,
        data_loader=load_data):
    args = args_parser()
    if args.seed:
        # Use a seed for deterministic results
        # random.seed(args.seed)
        np.random.seed(int(args.seed))
    logging.basicConfig(format=log_format, level=_log_level(args.verbose))
    return data_loader(args.arff)
