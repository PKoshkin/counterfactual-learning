from utils import read_csv, write_to_csv, calculate_metric

import numpy as np
import os
import pytest


def test_read_write():
    test_file_name = '_test_read_write.csv'
    pool = np.random.rand(12000, 10)
    write_to_csv(pool, test_file_name)
    readed_pool = read_csv(test_file_name)
    os.remove(test_file_name)

    assert readed_pool.shape == pool.shape, "Source and readed pools have different shapes!"

    matched_elems = np.sum(np.abs(readed_pool - pool) < 1e-10)
    not_match_error = "Source and readed pools have diffenent elements!"
    assert matched_elems == (pool.shape[0] * pool.shape[1]), not_match_error


PROBA = 1.0 / 11
BIG_PROBA = 100000

POSITIONS = [
    [2, 3, 1, 4, 5, 0],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2] * 11,
]
POOLS = [
    np.array([
        [1, PROBA, 1, 0],
        [2, PROBA, 3, 0],
        [1, PROBA, 0, 0],
        [1, PROBA, 0, 0],
        [10, BIG_PROBA, 5, 0],
        [1, 1, 0, 1],
    ]),
    np.array([
        [1, PROBA, 2, 0],
        [1, PROBA, 1, 0],
        [1, PROBA, 1, 0],
        [1, PROBA, 1, 0],
    ]),
    np.array([
        [1, PROBA, 0, 0],
        [1, PROBA, 1, 0],
        [1, PROBA, 1, 0],
        [1, PROBA, 1, 0],
    ]),
    np.array([[1, PROBA, 1, 0]] * 10 + [[2, PROBA, 2, 0]]),
]
RIGHTS = [
    (2 / PROBA + 10 / BIG_PROBA + 1.0 / 1) / 6,
    1 / PROBA / 4,
    0,
    2,
]


@pytest.mark.filterwarnings('ignore: invalid value encountered in double_scalars')
def test_metric_normed():
    denominators = [
        (1 / PROBA + 1 / BIG_PROBA + 1) / 6,
        1 / PROBA / 4,
        0,
        1,
    ]

    for pos, pool, right, denominator in zip(POSITIONS, POOLS, RIGHTS, denominators):
        normed_metric = calculate_metric(pos, pool, True)
        if denominator == 0:
            assert np.isnan(normed_metric)
        else:
            assert abs(normed_metric - right / denominator) < 1e-4


def test_metric_unnormed():
    for pos, pool, right in zip(POSITIONS, POOLS, RIGHTS):
        metric = calculate_metric(pos, pool, False)
        assert abs(metric - right) < 1e-4
