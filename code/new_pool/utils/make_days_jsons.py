from __future__ import print_function
import argparse
import os
from datetime import datetime
import time
import random
from json import loads as json_from_string, dumps


def make_days_jsons(pool_path, skip_prob, out_folder, timestamps, debug):
    handlers = {
        timestamp: open(os.path.join(out_folder, "day_{}.json".format(i)), 'w')
        for i, timestamp in enumerate(timestamps)
    }
    # read pool by pool_path
        if random.random() > skip_prob:
            timestamp = int(item["ts"])
            date = datetime.fromtimestamp(timestamp).date()
            day_timestamp = int(time.mktime(date.timetuple()))
            print(dumps(item), file=handlers[day_timestamp])
    if debug:
        i = 5
        length = len([
            line for line in handlers[timestamps[i]]
        ])
        for pos in range(10) + [100]:
            probs = sum([
                1 / float(json_from_string(line.strip())['p'])
                for line in handlers[timestamps[i]]
                if json_from_string(line.strip())['pos'] == pos
            ])
            print("pos={}, sum(1/p)={}, len={}", pos, probs, length)


def main():
    timestamps = [
        1527541200,
        1527714000,
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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    make_days_jsons(args.pool_path, args.skip_prob, args.out_folder, timestamps, args.debug)


if __name__ == "__main__":
    main()
