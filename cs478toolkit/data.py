import arff
import numpy as np
import re

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

    return _split(data, label_size)d