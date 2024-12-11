import numpy as np
from copy import deepcopy


np.random.seed(30)


def reduce_data(data_set, threshold=0.01):
    """ Returns the reduced dataset using variance thresholding

    Args:
        data_set (ndarray(int, ndarray)): processed data
        threshold (float): variance threshold, default 0.01

    Returns:
        tuple of (ndarray(int, ndarray), list): Reduced dataset and removed features
    """
    data_cp = deepcopy(data_set)
    features = np.array([feature[1] for feature in data_cp])
    variances = np.var(features, axis=0)
    removed_features = [index for index, variance in enumerate(
        variances) if variance < threshold]

    for entry in data_cp:
        entry[1] = np.delete(entry[1], removed_features)

    return data_cp, removed_features


def reduce_query(data_set, removed_features):
    """ Returns the reduced query point

    Args:
        (int, ndarray): image
        removed_features (list): list of removed features

    Returns:
        (int, ndarray): Reduced image
    """
    query_cp = deepcopy(data_set)
    for entry in query_cp:
        entry[1] = np.delete(entry[1], removed_features)

    return query_cp


def read_data(file_name: str) -> list:

    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label, np.array(attribs, dtype=float)])
    return (data_set)


def show(file_name, mode):

    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ', end='')
                else:
                    print('*', end='')
            else:
                print('%4s ' % data_set[obs][1][idx], end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0], end='')
        print(' ')
