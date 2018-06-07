from __future__ import print_function
import argparse
import numpy as np
import os
import sys
sys.path.append("../utils")
from json_tools import get_from_pool
from constants import DAYS_NUMBER
from metric import calculate_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    args = parser.parse_args()
    filenames = [
        os.path.join(args.data_folder, "day_{}.json".format(i))
        for i in range(1, DAYS_NUMBER)
    ]
    positions = list(range(10))
    out_files = [open(os.path.join(args.out_folder, "pos_{}.txt".format(pos)), "w") for pos in positions]
    for pos in range(10):
        for out_file, filename in zip(out_files, filenames):
            probs = get_from_pool(filename, "p", float)
            positions = get_from_pool(filename, "pos", int)
            targets = get_from_pool(filename, "target", int)
            constant_predictions = np.array([pos] * len(probs))
            metric = calculate_metric(constant_predictions, positions, targets, probs)
            print(metric, file=out_file)


if __name__ == "__main__":
    main()
