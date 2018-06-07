from __future__ import print_function
import argparse
import os
from datetime import datetime
import time
import random
from pool_iterator import pool_iterator
from json import dumps


def make_days_jsons(pool_iterator, skip_prob, out_folder, timestamps, seed):
    random.seed(seed)
    handlers = {
        timestamp: open(os.path.join(out_folder, "day_{}.json".format(i)), 'w')
        for i, timestamp in enumerate(timestamps)
    }
    # read pool by pool_iterator
    for item in pool_iterator():
        if random.random() > skip_prob:
            timestamp = int(item["ts"])
            date = datetime.fromtimestamp(timestamp).date()
            day_timestamp = int(time.mktime(date.timetuple()))

            print(dumps(item), file=handlers[day_timestamp])


def main():
    timestamps = [
        1527541200,
        1527627600,
        1527714000,
        1527800400,
        1527886800,
        1527973200,
        1528059600,
    ]
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
        timestamps,
        args.seed
    )


if __name__ == "__main__":
    main()
