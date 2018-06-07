import numpy as np
import sys
sys.path.append("../utils")


def get_from_pool(pool_iterator, name, constructor=float):
    return np.array([constructor(item[name]) for item in pool_iterator])


def get_binary_labels(pool_iterator, trashhold):
    return get_from_pool(pool_iterator, "target", lambda item: 1 if item > trashhold else 0)


def get_regression_labels(pool_iterator):
    return get_from_pool(pool_iterator, "target", float)


def get_classification_labels(pool_iterator, max_clicks):
    # if max_clicks=3 classes are: [0 clicks, 1 click, 2 clicks, 3 clisck, more then 3 clicks]
    return get_from_pool(pool_iterator, "target", lambda item: min(int(item), max_clicks + 1))


def make_feature(json, add_positions, first_feature, last_feature):
    if add_positions:
        return [json["pos"]] + json["factors"][first_feature:last_feature]
    else:
        return json["factors"][first_feature:last_feature]


def get_linear_stacked_features(pool_iterator, results_list, add_positions=True):
    """
    results_list: list of strings - filenames of files with linear models predictions
    """
    results = [np.load(filename) for filename in results_list]
    features = []
    for i, item in enumerate(pool_iterator):
        feature = make_feature(item, add_positions, 0, -1)
        for result in results:
            # take zero prediction (pos=0) to ignore position
            feature.append(result[i][0])
        features.append(feature)
    return np.array(features)


def get_features_range(pool_iterator, first_feature=0, last_feature=-1, add_positions=True):
    """
    Takes features from pool_iterator with indices from first_feature to last_feature and.
    If add_positions == True concatinates position as zero feature. Index last_feature is not included.
    """
    assert first_feature >= 0
    if last_feature != -1:
        assert first_feature < last_feature
    assert type(add_positions) == bool
    features = []
    for item in pool_iterator:
        features.append(make_feature(item, add_positions, first_feature, last_feature))
    return np.array(features)


def get_features(pool_iterator, add_positions=True):
    return get_features_range(pool_iterator, 0, -1, add_positions)


def get_labels(pool_iterator):
    return get_from_pool(pool_iterator, "target", int)
