from __future__ import print_function
import os
import sys
import time
import numpy as np

sys.path.append("../utils")
from constants import DAYS_NUMBER
from json_tools import get_features, get_binary_labels


def calculate_binary_classification_predictions(model_constructor, data_folder, out_folder, threshold):
    """
    model_constructor: regression model. Has fit(x, y) and predict(x) methods.
    data_folder: str. Directory, containing files "day_i.json" where i in range(DAYS_NUMBER).
    out_folder: str. Directory, to save results. DAYS_NUMBER - 1 files will be created.
    """
    # features contain positions
    json_filenames = [os.path.join(data_folder, "day_{}.json".format(i)) for i in xrange(DAYS_NUMBER)]
    features = [get_features(json_filename, False) for json_filename in json_filenames]
    labels = [get_binary_labels(json_filename, threshold) for json_filename in json_filenames]
    model = model_constructor()

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
                predictions = model.predict_proba(features[i])
                np.save(res_handler, predictions)
                print(predictions, file=res_handler)
                end = time.time()
                predict_time = end - start
                print(train_time, predict_time, file=times_handler)
