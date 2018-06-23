from __future__ import print_function
import os
import sys
import numpy as np
import pickle

sys.path.append("../utils")
from constants import POSITIONS_VARIANTS, DAYS_NUMBER
from json_tools import get_day_features
from log import log


def make_predictions(args):
    """
    args contain:
        model_path: str. path to model to test
        verbose: bool. Wether to print logs to stdout or not.
        out_folder: str. Directory, to save results. 1 file will be created.
        type: str. One of ["regression", "classification", "binary_classification"]
        position_features_num: int
        test_days: list of ints. Numbers of days to test on.

        if args.tpye == "classification"
            args.max_clicks should be provided
    """
    assert args.type in ["classification", "regression", "binary_classification"]

    json_filenames = [os.path.join(args.data_folder, "day_{}.json".format(day)) for day in range(DAYS_NUMBER)]
    test_features = [
        get_day_features(json_filenames[test_day], test_day, args, args.position_features_num > 0)
        for test_day in args.test_days
    ]
    model = pickle.load(open(args.model_path, "rb"))

    if args.verbose:
        log("days to predict: {}".format(args.test_days))
        log("loaded model from: {}".format(args.model_path))

    for features, test_day in zip(test_features, args.test_days):
        if args.verbose:
            log("features shape: {}".format(np.shape(features)))
            log("start predicting on day {}".format(test_day))
        if args.type.endswith("classification"):
            predictions = model.predict_proba(features)
        else:
            predictions = model.predict(features)

        if args.position_features_num > 0:
            if args.type == "classification":
                predictions = np.reshape(predictions, [-1, len(POSITIONS_VARIANTS), args.max_clicks + 2])
            elif args.type == "regression":
                predictions = np.reshape(predictions, [-1, len(POSITIONS_VARIANTS)])
            else:
                predictions = np.reshape(predictions, [-1, len(POSITIONS_VARIANTS), 2])

        index = args.model_path.find("_trained_on_")
        assert index != -1
        index += len("_trained_on_")
        train_days = args.model_path[index:]
        filename = os.path.join(
            args.out_folder,
            "train_{}_test_{}".format(train_days, test_day)
        )
        if args.verbose:
            log("predictions shape: {}".format(np.shape(predictions)))
            log("saveing results to {}".format(filename))
        np.save(filename, np.array(predictions))
        if args.verbose:
            log("results saved")
