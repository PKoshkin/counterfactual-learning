#!/usr/bin/env python2.7

from table_iterator import table_iterator
import argparse
import pandas as pd
import numpy as np
import sys
import random
import time
from utils import write_to_csv, log
from math import ceil
from collections import namedtuple


def _get_row_and_date(item, date_format):
    target = int(item['target'])
    prob = float(item['p'])

    ts = int(item['ts'])
    date = time.strftime(date_format, time.localtime(ts))

    position = item.get('pos')
    if position is None:
        position = item['rnd_pos']

    return date, [target, prob, position] + item['factors']


class Pool(object):
    def __init__(self, path, skip_prob, chunk_size):
        self._path = path
        self._skip_prob = skip_prob
        self._chunk = []
        self._chunk_size = chunk_size
        self._size = 0

    def add_row_with_prob(self, row):
        if self._skip_prob < random.random():
            self._chunk.append(row)
        if len(self._chunk) >= self._chunk_size:
            self.upload_current_chuck()

    def upload_current_chuck(self):
        write_to_csv(self._chunk, self._path, mode='a')
        self._size += len(self._chunk)
        self._chunk = []

    @property
    def size(self):
        return self._size


def create_pools(args):
    pools_dict = {
        date: Pool(path, skip_prob, args.chunk_size)
        for date, path, skip_prob in zip(
            args.output_dates,
            args.output_paths,
            args.output_skip_probs
        )
    }

    for item_ind, item in enumerate(table_iterator(args.input_pool_path)):
        date, row = _get_row_and_date(item, args.date_format)
        pool = pools_dict.get(date)
        if pool is not None:
            pool.add_row_with_prob(row)

    for date, pool in pools_dict.items():
        pool.upload_current_chuck()
        pool_ind = args.output_dates.index(date)
        log('pool {:2} for date {:10} size: {}\n'.format(pool_ind, date, pool.size))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pool_path', required=True)
    parser.add_argument('--output_paths', required=True, nargs='+')
    parser.add_argument('--output_dates', required=True, nargs='+')
    parser.add_argument('--output_skip_probs', required=True, type=float, nargs='+')
    parser.add_argument('--date_format', default="%Y-%m-%d")
    parser.add_argument('--chunk_size', default=100, type=int)
    args = parser.parse_args()

    paths_and_dates = len(args.output_paths) != len(args.output_dates)
    paths_and_skip_probs = len(args.output_paths) != len(args.output_skip_probs)
    if paths_and_dates or paths_and_skip_probs:
        parser.error('len of output_paths, output_dates and output_skip_probs should be equal')
    return args


def main():
    args = parse_args()
    create_pools(args)


if __name__ == '__main__':
    main()
