from __future__ import print_function
import numpy as np
import sys
import os

sys.path.append("../utils")
from metric import calculate_metric
from json_tools import get_from_pool
from pool_iterator import pool_iterator
from log import log
from constants import NONE_POSITION


def evaluate(predictions_folder,
             data_folder,
             out_folder,
             positions_by_predictions,
             predictios_masks=None,
             verbose=False):
    predictions_filenames = sorted([
        os.path.join(predictions_folder, filename)
        for filename in os.listdir(predictions_folder)
    ])
    if verbose:
        log("predictions_filenames: {}".format(predictions_filenames))
    # prediction filename is like "train_i_test_j" where i and j are int days numbers (hope their len is 1)
    days = sorted([int(filename[-5]) for filename in predictions_filenames])
    if verbose:
        log("will evaluate days: {}".format(days))
    data_filenames = [
        os.path.join(data_folder, "day_{}.json".format(day))
        for day in days
    ]
    with open(os.path.join(out_folder, "metrics.txt"), "w") as handler:
        for i, (predictions_filename, data_filename) in enumerate(zip(predictions_filenames, data_filenames)):
            if verbose:
                log("predict on file \"{}\"".format(predictions_filename))
            predictions = np.load(predictions_filename)
            predicted_positions = positions_by_predictions(predictions)
            if predictios_masks is not None:
                predicted_positions[(1 - predictios_masks[i]).astype(bool)] = NONE_POSITION
            probs = get_from_pool(pool_iterator(data_filename), "p", float)
            target_positions = get_from_pool(pool_iterator(data_filename), "pos", int)
            targets = get_from_pool(pool_iterator(data_filename), "target", int)

            if verbose:
                log("calculating metric")
            metric = calculate_metric(predicted_positions, target_positions, targets, probs)
            counter = len(target_positions[target_positions == predicted_positions])
            print(metric, counter, file=handler)
