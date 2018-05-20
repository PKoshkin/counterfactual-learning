from __future__ import print_function
import numpy as np
import sys
import os

sys.path.append("../utils")
from metric import calculate_metric
from json_tools import get_from_pool
from constants import DAYS_NUMBER


def evaluate(predictions_folder, data_folder, out_folder, positions_by_predictions):
    predictions_filenames = [
        os.path.join(predictions_folder, '_'.join(map(str, range(i))) + '-' + str(i) + '.txt')
        for i in range(1, DAYS_NUMBER)
    ]
    data_filenames = [
        os.path.join(data_folder, "day_{}.json".format(i))
        for i in range(1, DAYS_NUMBER)
    ]
    with open(os.path.join(out_folder, "metrics.txt"), "w") as handler:
        for predictions_filename, data_filename in zip(predictions_filenames, data_filenames):
            predictions = np.load(predictions_filename)
            predicted_positions = positions_by_predictions(predictions)
            probs = get_from_pool(data_filename, "p", float)
            target_positions = get_from_pool(data_filename, "pos", int)
            targets = get_from_pool(data_filename, "target", int)

            metric = calculate_metric(predicted_positions, target_positions, targets, probs)
            print(metric, file=handler)
