from __future__ import print_function
import argparse
import os
from datetime import datetime
import time
import random
from pool_iterator import pool_iterator
from constants import TIMESTAMPS
from json import dumps


def make_days_jsons(pool_iterator, skip_prob, out_folder, seed):
    random.seed(seed)
    handlers = {
        timestamp: open(os.path.join(out_folder, "day_{}.json".format(i)), 'w')
        for i, timestamp in enumerate(TIMESTAMPS)
    }
    # read pool by pool_iterator
    for item in pool_iterator():
        if random.random() > skip_prob:
            timestamp = int(item["ts"])
            date = datetime.fromtimestamp(timestamp).date()
            day_timestamp = int(time.mktime(date.timetuple()))

            print(dumps(item), file=handlers[day_timestamp])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool_path", required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--skip_prob", type=float, required=True)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()

    make_days_jsons(
        lambda: pool_iterator(args.pool_path),
        args.skip_prob,
        args.out_folder,
        args.seed
    )


if __name__ == "__main__":
    main()
