from __future__ import print_function
import os
import sys
import numpy as np
from catboost import Pool

sys.path.append("../utils")
from constants import POSITIONS_VARIANTS, DAYS_NUMBER
from json_tools import get_linear_stacked_features, get_labels
from pool_iterator import pool_iterator
from log import log


def calculate_predictions(args):
    """
    args contain:
        data_folder: str. Directory, containing files "day_i.json" where i in range(DAYS_NUMBER).
        out_folder: str. Directory, to save results. 1 file will be created.
        type: str. One of ["regression", "classification", "binary_classification", "binary_regression"]
        model_constructor: callable. Takes verbose param. If type == "classification" also takes max_clicks param.
        additional_features: (dict day -> additional features for day) or None.
        labels_to_substruct: (dict day -> targets to substruct for day) or None.
        train_days: list of ints. Numbers of days to train on.
        test_days: list of ints. Numbers of days to test on.
        validation_day: int or None. Number of day to validate on.
    """
    log("preprocesing started")

    if args.type == "classification":
        model = args.model_constructor(args.verbose, args.max_clicks)
    else:
        model = args.model_constructor(args.verbose)

    json_filenames = [os.path.join(args.data_folder, "day_{}.json".format(day)) for day in range(DAYS_NUMBER)]

    need_position_feature = not args.type.startswith("binary")

    def get_day_features(day):
        if args.additional_features is not None:
            return get_linear_stacked_features(
                pool_iterator(json_filenames[day]),
                args.additional_features[day],
                first_feature=args.first_feature,
                last_feature=args.last_feature,
                add_positions=need_position_feature
            )
        else:
            return get_linear_stacked_features(
                pool_iterator(json_filenames[day]),
                first_feature=args.first_feature,
                last_feature=args.last_feature,
                add_positions=need_position_feature
            )

    train_features = np.concatenate([get_day_features(day) for day in args.train_days], axis=0)
    test_features = [get_day_features(test_day) for test_day in args.test_days]
    log("train features shape: {}".format(np.shape(train_features)))

    def get_day_labels(day):
        if args.labels_to_substruct is not None:
            labels_to_substruct = np.load(args.labels_to_substruct[day])
            day_labels = get_labels(pool_iterator(json_filenames[day]), args)
            assert np.shape(labels_to_substruct) == np.shape(day_labels)
            return day_labels - labels_to_substruct
        else:
            return get_labels(pool_iterator(json_filenames[day]), args)

    train_labels = np.concatenate([get_day_labels(day) for day in args.train_days], axis=0)

    if args.validation_day is not None:
        validation_pool = Pool(get_day_features(args.validation_day), get_day_labels(args.validation_day))

    log("preprocesing finished")
    log("start training on days {}".format(args.train_days))

    if args.validation_day is not None:
        log("using fit with validation")
        model.set_params(iterations=2000)
        model.fit(
            train_features,
            train_labels,
            eval_set=validation_pool,
            use_best_model=True
        )
    else:
        log("using fit without validation")
        model.fit(train_features, train_labels)

    if "tree_count_" in dir(model):
        log("built {} trees".format(model.tree_count_))

    reshaped_positions = np.reshape(np.array(POSITIONS_VARIANTS), [-1, 1])
    for i, test_day in enumerate(args.test_days):
        log("start predicting on day {}".format(test_day))
        if args.type == "binary_classification":
            predictions = model.predict_proba(test_features[i])
        elif args.type == "binary_regression":
            predictions = model.predict(test_features[i])
        else:
            predictions = []
            for feature in test_features[i]:
                repeated_feature = np.repeat(np.array([feature[1:]]), len(POSITIONS_VARIANTS), axis=0)
                features_to_predict = np.concatenate([reshaped_positions, repeated_feature], axis=1)
                if args.type == "regression":
                    current_predictions = model.predict(features_to_predict)
                else:
                    current_predictions = model.predict_proba(features_to_predict)
                predictions.append(current_predictions)

        log("saveing results")
        np.save(
            os.path.join(args.out_folder, "train_{}_test_{}".format("_".join(map(str, args.train_days)), test_day)),
            np.array(predictions)
        )
        log("results saved")
