from __future__ import print_function
import os
import sys
import numpy as np

sys.path.append("../utils")
from constants import POSITIONS_VARIANTS, DAYS_NUMBER
from json_tools import get_linear_stacked_features, get_labels
from pool_iterator import pool_iterator


def calculate_predictions(args):
    """
    args contain:
        data_folder: str. Directory, containing files "day_i.json" where i in range(DAYS_NUMBER).
        out_folder: str. Directory, to save results. DAYS_NUMBER - 1 files will be created.
        type: str. One of ["regression", "classification", "binary_classification", "binary_regression"]
        additional_features: list of additional features for all days
    """
    if args.type == "classification":
        model = args.model_constructor(args.verbose, args.max_clicks)
    else:
        model = args.model_constructor(args.verbose)

    has_additionsl_features = args.additional_features is not None
    days_range = range(1, DAYS_NUMBER) if has_additionsl_features else range(DAYS_NUMBER)
    json_filenames = sorted(
        [os.path.join(args.data_folder, "day_{}.json".format(day)) for day in days_range],
    )

    need_position_feature = (not args.type.startswith("binary"))
    if has_additionsl_features:
        features = [
            get_linear_stacked_features(
                pool_iterator(json_filename),
                linear_prediction,
                args.first_feature,
                args.last_feature,
                need_position_feature
            ) for linear_prediction, json_filename in zip(args.additional_features, json_filenames)
        ]
    else:
        features = [
            get_linear_stacked_features(
                pool_iterator(json_filename),
                first_feature=args.first_feature,
                last_feature=args.last_feature,
                add_positions=need_position_feature
            ) for json_filename in json_filenames
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
            elif args.type == "binary_regression":
                predictions = model.predict(features[i])
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
