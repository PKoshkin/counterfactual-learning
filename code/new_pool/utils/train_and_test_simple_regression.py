from __future__ import print_function
import os
import time
import numpy as np

from metric import calculate_metric
from constants import DAYS_NUMBER, POSITIONS_NUMBER
from json_tools import get_features, get_labels, get_from_pool


def predict_positions(model, features):
    """
    features: np.array - constants positions on index 0, shape: [nlines, nfeaatures + 1]
    """
    predicted_positions = []
    features_without_positions = features[:, 1:]
    positions = np.reshape(np.arange(POSITIONS_NUMBER), [POSITIONS_NUMBER, 1])
    for feature in features_without_positions:
        repeated_feature = np.repeat(np.array([feature]), POSITIONS_NUMBER, axis=0)
        features_to_predict = np.concatenate([positions, repeated_feature], axis=1)
        predictions = model.predict(features_to_predict)
        predicted_positions.append(np.argmax(predictions))
    return np.array(predicted_positions)


def train_and_test_simple_regression(model_constructor, data_folder, out_folder):
    """
    data_folder: directory name, containing files "day_i.json" where i in range(DAYS_NUMBER)
    """
    # features contain positions
    json_filenames = [os.path.join(data_folder, "day_{}.json".format(i)) for i in xrange(DAYS_NUMBER)]
    features = [get_features(json_filename, True) for json_filename in json_filenames]
    labels = [get_labels(json_filename) for json_filename in json_filenames]
    target_positions = [get_from_pool(json_filename, "pos") for json_filename in json_filenames]
    targets = [get_from_pool(json_filename, "target") for json_filename in json_filenames]
    probas = [get_from_pool(json_filename, "p") for json_filename in json_filenames]
    models = [model_constructor() for _ in xrange(DAYS_NUMBER - 1)]
    trains, tests = [], []
    for i in xrange(1, DAYS_NUMBER):
        trains.append(range(i))
        tests.append(i)

    with open(os.path.join(out_folder, "times.txt"), 'w') as times_handler,\
         open(os.path.join(out_folder, "metrics.txt"), 'w') as metrics_handler:
        for i, (model, train, test) in enumerate(zip(models, trains, tests)):
            start = time.time()
            for train_index in train:
                model.fit(features[train_index], labels[train_index], verbose=True)
            end = time.time()
            train_time = end - start

            start = time.time()
            predicted_positions = predict_positions(model, features[test])
            metric = calculate_metric(predicted_positions, target_positions[test], targets[test], probas[test])
            end = time.time()
            predict_time = end - start
            print(metric, file=metrics_handler)
            print(train_time, predict_time, file=times_handler)
