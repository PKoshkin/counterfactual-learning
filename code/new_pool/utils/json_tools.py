import numpy as np
from json import loads as json_from_string


def get_binary_labels(json_filename, trashhold):
    with open(json_filename) as handler:
        return np.array([
            1 if (json_from_string(line)["target"] > trashhold) else 0
            for line in handler
        ])


def get_from_pool(json_filename, name):
    with open(json_filename) as handler:
        return np.array([
            json_from_string(line)[name]
            for line in handler
        ])


def get_regression_labels(json_filename):
    return map(float, get_from_pool(json_filename, "target"))


def get_classification_labels(json_filename, max_clicks):
    # if max_clicks=3 classes are: [0 clicks, 1 click, 2 clicks, 3 clisck, more then 3 clicks]
    real_clicks = map(int, get_from_pool(json_filename, "target"))
    return map(lambda x: min(x, max_clicks + 1), real_clicks)


def make_feature(json, add_positions):
    if add_positions:
        return [json["pos"]] + json["factors"]
    else:
        return json["factors"]


def get_features(json_filename, add_positions=True):
    assert type(add_positions) == bool
    with open(json_filename) as handler:
        features = []
        for line in handler:
            json = json_from_string(line)
            features.append(make_feature(json, add_positions))
    return np.array(features)


def get_labels(json_filename):
    return map(int, get_from_pool(json_filename, "target"))
