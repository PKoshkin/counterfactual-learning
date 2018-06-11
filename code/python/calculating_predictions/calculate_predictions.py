from __future__ import print_function
import os
import sys
import numpy as np

sys.path.append("../utils")
from constants import POSITIONS_VARIANTS
from json_tools import get_features_range, get_labels
from pool_iterator import pool_iterator


def calculate_predictions(args):
    """
    args contain:
        data_folder: str. Directory, containing files "day_i.json" where i in range(DAYS_NUMBER).
        out_folder: str. Directory, to save results. DAYS_NUMBER - 1 files will be created.
        type: str. One of ["regression", "classification", "binary_classification"]
    """
    if args.type == "classification":
        model = args.model_constructor(args.verbose, args.max_clicks)
    elif args.type == "binary_classification":
        model = args.model_constructor(args.verbose)
    elif args.type == "regression":
        model = args.model_constructor(args.verbose)
    else:
        raise ValueError("Wrong type \"{}\"".format(args.type))

    json_filenames = sorted(
        [os.path.join(args.data_folder, filename) for filename in os.listdir(args.data_folder)],
        key=lambda filename: int(filename[-6])
    )

    need_position_feature = (args.type == "binary_classification")
    features = [
        get_features_range(pool_iterator(json_filename), args.first_feature, args.last_feature, need_position_feature)
        for json_filename in json_filenames
    ]
    labels = [get_labels(pool_iterator(json_filename), args) for json_filename in json_filenames]

    reshaped_positions = np.reshape(np.array(POSITIONS_VARIANTS), [-1, 1])

    for i in range(1, len(json_filenames)):
        # i - index of test, (i-1) - index of train
        res_filename = "train_{}_test_{}.txt".format(i - 1, i)
        with open(os.path.join(args.out_folder, res_filename), 'w') as res_handler:
            model.fit(features[i - 1], labels[i - 1])

            if args.type == "binary_classification":
                predictions = model.predict_proba(features[i])
            else:
                predictions = []
                for feature in features[i]:
                    repeated_feature = np.repeat(np.array([feature[1:]]), len(POSITIONS_VARIANTS), axis=0)
                    features_to_predict = np.concatenate([reshaped_positions, repeated_feature], axis=1)
                    if args.type == "regression":
                        current_predictions = model.predict(features_to_predict)
                    else:
                        current_predictions = model.predict_proba(features_to_predict)
                    predictions.append(current_predictions)
            np.save(res_handler, np.array(predictions))
