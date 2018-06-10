from __future__ import print_function
import os
import sys
import time
import numpy as np

sys.path.append("../utils")
from constants import DAYS_NUMBER, POSITIONS_VARIANTS
from json_tools import get_linear_stacked_features, get_classification_labels
from pool_iterator import pool_iterator


def calculate_classification_stacked_on_linear_predictions(args):
    """
    model_constructor: regression model. Has fit(x, y) and predict(x) methods.
    data_folder: str. Directory, containing files "day_i.json" where i in range(DAYS_NUMBER).
    out_folder: str. Directory, to save results. DAYS_NUMBER - 1 files will be created.
    linear_predictions: list of len linear_models_number with linear models results folders
    """


    real_linear_predictions = []
    for linear_prediction in args.linear_predictions:
        predictions_folders = [
            os.path.join(linear_prediction, filename)
            for filename in os.listdir(linear_prediction)
        ]
        real_linear_predictions.extend(predictions_folders)
    args.linear_predictions = real_linear_predictions

    linear_predictions_for_days = {day: [] for day in range(DAYS_NUMBER - 1)}
    for day in range(DAYS_NUMBER - 1):
        for results_folder in args.linear_predictions:
            # linear predictions do not exist for the first day
            res_filename = '_'.join(map(str, range(day + 1))) + '-' + str(day + 1) + '.txt'
            linear_predictions_for_days[day].append(os.path.join(results_folder, res_filename))

    json_filenames = [os.path.join(args.data_folder, "day_{}.json".format(i)) for i in xrange(1, DAYS_NUMBER)]
    # features contain positions
    features = [
        get_linear_stacked_features(pool_iterator(json_filename), linear_predictions_for_days[day], True)
        for day, json_filename in enumerate(json_filenames)
    ]
    labels = [get_classification_labels(pool_iterator(json_filename), args.max_clicks) for json_filename in json_filenames]
    model = args.model_constructor(args.verbose, args.max_clicks)

    reshaped_positions = np.reshape(np.array(POSITIONS_VARIANTS), [-1, 1])

    with open(os.path.join(args.out_folder, "times.txt"), 'w') as times_handler:
        for i in range(1, DAYS_NUMBER - 1):
            # i - index of test, (i-1) - index of train (indices in features array).
            # real days indices are (i + 1) for test and i for train
            res_filename = '_'.join(map(str, range(1, i + 1))) + '-' + str(i + 1) + '.txt'
            with open(os.path.join(args.out_folder, res_filename), 'w') as res_handler:
                start = time.time()
                model.fit(features[i - 1], labels[i - 1])
                end = time.time()
                train_time = end - start

                all_predictions = []
                start = time.time()
                # features
                for feature in features[i]:
                    repeated_feature = np.repeat(np.array([feature[1:]]), len(POSITIONS_VARIANTS), axis=0)
                    features_to_predict = np.concatenate([reshaped_positions, repeated_feature], axis=1)
                    probas_predictions = model.predict_proba(features_to_predict)
                    all_predictions.append(probas_predictions)
                np.save(res_handler, np.array(all_predictions))

                end = time.time()
                predict_time = end - start
                print(train_time, predict_time, file=times_handler)
