import numpy as np
from json import loads as json_from_string


def get_from_pool(json_filename, name):
    with open(json_filename) as handler:
        return np.array([
            json_from_string(line)[name]
            for line in handler
        ])


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
    return get_from_pool(json_filename, "target")
