from __future__ import print_function
import os
import sys
import time
import numpy as np

sys.path.append("../utils")
from constants import DAYS_NUMBER, POSITIONS_NUMBER, NONE_POSITION
from json_tools import get_features, get_labels


def calculate_simple_classification_predictions(model_constructor, data_folder, out_folder):
    """
    model_constructor: regression model. Has fit(x, y) and predict(x) methods.
    data_folder: str. Directory, containing files "day_i.json" where i in range(DAYS_NUMBER).
    out_folder: str. Directory, to save results. DAYS_NUMBER - 1 files will be created.
    """
    # features contain positions
    json_filenames = [os.path.join(data_folder, "day_{}.json".format(i)) for i in xrange(DAYS_NUMBER)]
    features = [get_features(json_filename, True) for json_filename in json_filenames]
    labels = [get_labels(json_filename) for json_filename in json_filenames]
    models = [model_constructor() for _ in xrange(DAYS_NUMBER - 1)]
    trains, tests = [], []
    for i in xrange(1, DAYS_NUMBER):
        trains.append(range(i))
        tests.append(i)

    possible_positions = range(POSITIONS_NUMBER) + [NONE_POSITION]
    reshaped_positions = np.reshape(np.array(possible_positions), [-1, 1])

    with open(os.path.join(out_folder, "times.txt"), 'w') as times_handler:
        for i, (model, train, test) in enumerate(zip(models, trains, tests)):
            res_filename = '_'.join(map(str, train)) + '-' + str(test) + '.txt'
            with open(os.path.join(out_folder, res_filename), 'w') as res_handler:
                start = time.time()
                for train_index in train:
                    model.fit(features[train_index], labels[train_index])
                end = time.time()
                train_time = end - start

                start = time.time()
                for feature in features[test]:
                    repeated_feature = np.repeat(np.array([feature[1:]]), POSITIONS_NUMBER + 1, axis=0)
                    features_to_predict = np.concatenate([reshaped_positions, repeated_feature], axis=1)
                    probas_predictions = model.predict_proba(features_to_predict)
                    probas_predictions = ', '.join(map(str, probas_predictions))
                    print("!!!!!!!!!!!!!!!!!")
                    print('[' + probas_predictions + ']', file=res_handler)

                end = time.time()
                predict_time = end - start
                print(train_time, predict_time, file=times_handler)
