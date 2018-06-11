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


def get_labels(pool_iterator, args):
    if args.type == "classification":
        return get_classification_labels(pool_iterator, args.max_clicks)
    elif args.type == "regression" or args.type == "binary_regression":
        return get_regression_labels(pool_iterator)
    elif args.type == "binary_classification":
        return get_binary_labels(pool_iterator, args.threshold)
    else:
        raise ValueError("Wrong type {}".format(args.type))


def make_feature(json, add_positions, first_feature, last_feature):
    if add_positions:
        return [json["pos"]] + json["factors"][first_feature:last_feature]
    else:
        return json["factors"][first_feature:last_feature]


def get_linear_stacked_features(pool_iterator, results_list, first_feature=0, last_feature=-1, add_positions=True):
    """
    results_list: list of strings - filenames of files with linear models predictions
    """
    assert first_feature >= 0
    if last_feature != -1:
        assert first_feature < last_feature
    assert type(add_positions) == bool

    results = [np.load(filename) for filename in results_list]
    features = []
    for i, item in enumerate(pool_iterator):
        feature = make_feature(item, add_positions, first_feature, last_feature)
        for result in results:
            # take zero prediction (pos=0) to ignore position
            feature.append(result[i])
        features.append(feature)
    return np.array(features)
