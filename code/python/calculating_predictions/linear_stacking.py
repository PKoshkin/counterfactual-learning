from __future__ import print_function
import os
import sys

sys.path.append("../utils")
from constants import DAYS_NUMBER
from calculate_predictions import calculate_predictions


def calculate_classification_stacked_on_linear_predictions(args):
    """
    args contain:
        linear_predictions: list with linear models results folders
    """

    linear_predictions = []
    for linear_prediction in args.linear_predictions:
        predictions_folders = [
            os.path.join(linear_prediction, filename)
            for filename in os.listdir(linear_prediction)
        ]
        linear_predictions.extend(predictions_folders)
    args.linear_predictions = linear_predictions

    linear_predictions = []
    # linear predictions do not exist for the first day
    for day in range(1, DAYS_NUMBER):
        day_linear_predictions = []
        for results_folder in args.linear_predictions:
            res_filename = "train_{}_test_{}.txt".format(day - 1, day)
            day_linear_predictions.append(os.path.join(results_folder, res_filename))
        linear_predictions.append(day_linear_predictions)

    args.additional_features = linear_predictions
    args.type = "classification"

    return calculate_predictions(args)
