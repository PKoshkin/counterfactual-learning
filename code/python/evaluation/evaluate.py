from __future__ import print_function
import numpy as np
import sys
import os

sys.path.append("../utils")
from metric import calculate_metric
from json_tools import get_from_pool
from pool_iterator import pool_iterator


def evaluate(predictions_folder, data_folder, out_folder, positions_by_predictions):
    predictions_filenames = sorted([
        os.path.join(predictions_folder, filename)
        for filename in os.listdir(predictions_folder)
        if filename != "times.txt"
    ])
    days = sorted([int(filename[-5]) for filename in predictions_filenames])
    data_filenames = [
        os.path.join(data_folder, "day_{}.json".format(day))
        for day in days
    ]
    with open(os.path.join(out_folder, "metrics.txt"), "w") as handler:
        for predictions_filename, data_filename in zip(predictions_filenames, data_filenames):
            predictions = np.load(predictions_filename)
            predicted_positions = positions_by_predictions(predictions)
            probs = get_from_pool(pool_iterator(data_filename), "p", float)
            target_positions = get_from_pool(pool_iterator(data_filename), "pos", int)
            targets = get_from_pool(pool_iterator(data_filename), "target", int)

            metric = calculate_metric(predicted_positions, target_positions, targets, probs)
            counter = len(target_positions[target_positions == predicted_positions])
            print(metric, counter, file=handler)
