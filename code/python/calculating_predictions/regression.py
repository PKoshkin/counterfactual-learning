from __future__ import print_function
import os
import sys
import time
import numpy as np

sys.path.append("../utils")
from constants import DAYS_NUMBER, POSITIONS_VARIANTS
from json_tools import get_features_range, get_regression_labels
from pool_iterator import pool_iterator


def calculate_regression_predictions(model_constructor, data_folder, out_folder, first_feature=0, last_feature=-1):
    """
    model_constructor: regression model. Has fit(x, y) and predict(x) methods.
    data_folder: str. Directory, containing files "day_i.json" where i in range(DAYS_NUMBER).
    out_folder: str. Directory, to save results. DAYS_NUMBER - 1 files will be created.
    """

    # features contain positions
    json_filenames = [os.path.join(data_folder, "day_{}.json".format(i)) for i in xrange(DAYS_NUMBER)]
    features = [
        get_features_range(pool_iterator(json_filename), first_feature, last_feature, True)
        for json_filename in json_filenames
    ]
    labels = [get_regression_labels(pool_iterator(json_filename)) for json_filename in json_filenames]
    model = model_constructor()

    reshaped_positions = np.reshape(np.array(POSITIONS_VARIANTS), [-1, 1])

    with open(os.path.join(out_folder, "times.txt"), 'w') as times_handler:
        for i in range(1, DAYS_NUMBER):
            # i - index of test, (i-1) - index of train
            res_filename = '_'.join(map(str, range(i))) + '-' + str(i) + '.txt'
            with open(os.path.join(out_folder, res_filename), 'w') as res_handler:
                start = time.time()
                model.fit(features[i - 1], labels[i - 1])
                end = time.time()
                train_time = end - start

                start = time.time()
                all_predictions = []
                for feature in features[i]:
                    repeated_feature = np.repeat(np.array([feature[1:]]), len(POSITIONS_VARIANTS), axis=0)
                    features_to_predict = np.concatenate([reshaped_positions, repeated_feature], axis=1)
                    predictions = model.predict(features_to_predict)
                    all_predictions.append(predictions)
                np.save(res_handler, np.array(all_predictions))
                end = time.time()
                predict_time = end - start
                print(train_time, predict_time, file=times_handler)
