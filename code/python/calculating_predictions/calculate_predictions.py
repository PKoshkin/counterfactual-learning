from __future__ import print_function
import os
import sys
import datetime
import numpy as np

sys.path.append("../utils")
from constants import POSITIONS_VARIANTS
from json_tools import get_linear_stacked_features, get_labels
from pool_iterator import pool_iterator


def calculate_predictions(args):
    """
    args contain:
        data_folder: str. Directory, containing files "day_i.json" where i in range(DAYS_NUMBER).
        out_folder: str. Directory, to save results. 1 file will be created.
        type: str. One of ["regression", "classification", "binary_classification", "binary_regression"]
        additional_features: list of additional features for train and test day or None
        train_day: int. Number of day to train on. Test will be preformed on test_day == (train_day + 1)
    """
    print("\"{}\": preprocesing started.".format(str(datetime.datetime.now())))

    if args.type == "classification":
        model = args.model_constructor(args.verbose, args.max_clicks)
    else:
        model = args.model_constructor(args.verbose)

    train_day = args.train_day
    test_day = train_day + 1
    train_json_filename = os.path.join(args.data_folder, "day_{}.json".format(train_day))
    test_json_filename = os.path.join(args.data_folder, "day_{}.json".format(test_day))

    json_filenames = [train_json_filename, test_json_filename]

    need_position_feature = (not args.type.startswith("binary"))
    if args.additional_features is not None:

        print("len(af):", len(args.additional_features))
        print("af:", args.additional_features)


        train_features, test_features = [
            get_linear_stacked_features(
                pool_iterator(json_filename),
                linear_prediction,
                args.first_feature,
                args.last_feature,
                need_position_feature
            ) for linear_prediction, json_filename in zip(args.additional_features, json_filenames)
        ]
    else:
        train_features, test_features = [
            get_linear_stacked_features(
                pool_iterator(json_filename),
                first_feature=args.first_feature,
                last_feature=args.last_feature,
                add_positions=need_position_feature
            ) for json_filename in json_filenames
        ]

    train_labels, test_labels = [get_labels(pool_iterator(json_filename), args) for json_filename in json_filenames]

    reshaped_positions = np.reshape(np.array(POSITIONS_VARIANTS), [-1, 1])

    print("\"{}\": preprocesing finished.".format(str(datetime.datetime.now())))

    # i - index of test, (i-1) - index of train
    res_filename = "train_{}_test_{}".format(train_day, test_day)
    with open(os.path.join(args.out_folder, res_filename), 'w') as res_handler:
        print("\"{}\": start training on day {}.".format(str(datetime.datetime.now()), train_day))
        model.fit(train_features, train_labels)

        print("\"{}\": start testing on day {}.".format(str(datetime.datetime.now()), test_day))
        if args.type == "binary_classification":
            predictions = model.predict_proba(test_features)
        elif args.type == "binary_regression":
            predictions = model.predict(test_features)
        else:
            predictions = []
            for feature in test_features:
                repeated_feature = np.repeat(np.array([feature[1:]]), len(POSITIONS_VARIANTS), axis=0)
                features_to_predict = np.concatenate([reshaped_positions, repeated_feature], axis=1)
                if args.type == "regression":
                    current_predictions = model.predict(features_to_predict)
                else:
                    current_predictions = model.predict_proba(features_to_predict)
                predictions.append(current_predictions)

        print("\"{}\": saveing results.".format(str(datetime.datetime.now())))
        np.save(res_handler, np.array(predictions))
