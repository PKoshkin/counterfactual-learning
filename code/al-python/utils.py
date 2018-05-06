# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


POSITIONS = range(10) + [100]


def read_csv(path):
    return pd.read_csv(path, index_col=False, header=None, sep='\t').values


def write_to_csv(data, path, mode='w'):
    if len(np.shape(data)) != 2:
        return

    lines = ['\t'.join(map(str, line)) for line in data]
    data_str = '\n'.join(lines) + '\n'
    with open(path, mode) as handler:
        handler.write(data_str)


def get_targets(pool):
    return pool[:, 0]


def get_weights(pool):
    return 1.0 / pool[:, 1]


def get_positions(pool):
    return pool[:, 2]


def get_features(pool, add_position=True):
    if add_position:
        ind = 2
    else:
        ind = 3
    return pool[:, ind:]


def calculate_metric(predictions, pool):
    real_positions = get_positions(pool)
    weights = get_weights(pool)
    targets = get_targets(pool)

    return np.mean(weights * targets * (predictions == real_positions))
