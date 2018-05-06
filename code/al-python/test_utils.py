from utils import read_csv, write_to_csv

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
