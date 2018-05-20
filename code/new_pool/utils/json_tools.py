import numpy as np
from json import loads as json_from_string
import sys
sys.path.append("../utils")


def get_binary_labels(json_filename, trashhold):
    with open(json_filename) as handler:
        return np.array([
            1 if (json_from_string(line)["target"] > trashhold) else 0
            for line in handler
        ])


def get_from_pool(json_filename, name, constructor=float):
    with open(json_filename) as handler:
        return np.array([
            constructor(json_from_string(line)[name])
            for line in handler
        ])


def get_regression_labels(json_filename):
    return map(float, get_from_pool(json_filename, "target"))


def get_classification_labels(json_filename, max_clicks):
    # if max_clicks=3 classes are: [0 clicks, 1 click, 2 clicks, 3 clisck, more then 3 clicks]
    real_clicks = map(int, get_from_pool(json_filename, "target"))
    return map(lambda x: min(x, max_clicks + 1), real_clicks)


def make_feature(json, add_positions, first_feature, last_feature):
    if add_positions:
        return [json["pos"]] + json["factors"][first_feature:last_feature]
    else:
        return json["factors"][first_feature:last_feature]


def get_linear_stacked_features(json_filename, results_list, add_positions=True):
    """
    results_list: list of strings - filenames of files with linear models predictions
    """
    results = [np.load(filename) for filename in results_list]
    with open(json_filename) as handler:
        features = []
        for i, line in enumerate(handler):
            json = json_from_string(line)
            feature = make_feature(json, add_positions, 0, -1)
            for result in results:
                # take zero prediction (pos=0) to ignore position
                feature.append(result[i][0])
            features.append(feature)
    return np.array(features)


def get_features_range(json_filename, first_feature=0, last_feature=-1, add_positions=True):
    """
    Takes features from json_filename with indices from first_feature to last_feature and.
    If add_positions == True concatinates position as zero feature. Index last_feature is not included.
    """
    assert first_feature >= 0
    if last_feature != -1:
        assert first_feature < last_feature
    assert type(add_positions) == bool
    with open(json_filename) as handler:
        features = []
        for line in handler:
            json = json_from_string(line)
            features.append(make_feature(json, add_positions, first_feature, last_feature))
    return np.array(features)


def get_features(json_filename, add_positions=True):
    return get_features_range(json_filename, 0, -1, add_positions)


def get_labels(json_filename):
    return map(int, get_from_pool(json_filename, "target"))
