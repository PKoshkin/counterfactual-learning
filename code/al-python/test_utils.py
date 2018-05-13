from utils import read_csv, write_to_csv, calculate_metric

import numpy as np
import os


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


def test_metric():
    proba = 1.0 / 11
    big_proba = 100000

    positions = [
        [2, 3, 1, 4, 5, 0],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [2] * 11,
    ]
    pools = [
        np.array([
            [1, proba, 1, 0],
            [2, proba, 3, 0],
            [1, proba, 0, 0],
            [1, proba, 0, 0],
            [10, big_proba, 5, 0],
            [1, 1, 0, 1],
        ]),
        np.array([
            [1, proba, 2, 0],
            [1, proba, 1, 0],
            [1, proba, 1, 0],
            [1, proba, 1, 0],
        ]),
        np.array([
            [1, proba, 0, 0],
            [1, proba, 1, 0],
            [1, proba, 1, 0],
            [1, proba, 1, 0],
        ]),
        np.array([[1, proba, 1, 0]] * 10 + [[1, proba, 2, 0]]),
    ]
    rights = [
        (2 / proba + 10 / big_proba + 1.0 / 1) / 6,
        1 / proba / 4,
        0,
        1
    ]

    for pos, pool, right in zip(positions, pools, rights):
        metric = calculate_metric(pos, pool)
        assert abs(metric - right) < 1e-4
