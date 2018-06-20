import numpy as np
import pickle
from log import log
from constants import POSITIONS_VARIANTS


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


def make_feature(json, add_positions, first_feature, last_feature, different_positions):
    if different_positions:
        assert add_positions
        return [
            [prediction] + json["factors"][first_feature:last_feature]
            for prediction in POSITIONS_VARIANTS
        ]
    elif add_positions:
        return [json["pos"]] + json["factors"][first_feature:last_feature]
    else:
        return json["factors"][first_feature:last_feature]


def get_linear_stacked_features(pool_iterator,
                                models_list=[],
                                first_feature=0,
                                last_feature=-1,
                                add_positions=True,
                                different_positions=False,
                                verbose=False):
    """
    models_list: list of strings - filenames of files with models to predict features
    """
    assert first_feature >= 0
    if last_feature != -1:
        assert first_feature < last_feature
    assert type(add_positions) == bool

    features = np.array([
        make_feature(item, add_positions, first_feature, last_feature, different_positions)
        for item in pool_iterator
    ])
    if different_positions:
        features = np.reshape(features, [-1, np.shape(features)[2]])
    if verbose:
        log("    base features shape: {}".format(np.shape(features)))
    predictions = []
    for filename in models_list:
        model = pickle.load(open(filename, "rb"))
        # filename has template:
        #    model_all_features_trained_on_days
        #    or
        #    model_features_from_x_to_y_trained_on_days
        if filename.find("from") == -1:
            features_to_predict = features
        else:
            from_index = filename.find("_from_")
            trained_index = filename.find("_trained_")
            to_index = filename.find("_to_")
            low = int(filename[(from_index + 6):to_index])
            hight = int(filename[(to_index + 4):trained_index])
            if add_positions:
                features_to_predict = np.concatenate([features[:, 0:1], features[:, (low + 1):(hight + 1)]], axis=1)
            else:
                features_to_predict = features[:, low:hight]

        prediction = model.predict(features_to_predict)
        predictions.append(np.reshape(prediction, [-1, 1]))

    result = np.concatenate([features] + predictions, axis=1)
    if verbose:
        log("    {} features added".format(len(predictions)))
        log("    result shape: {}".format(np.shape(result)))
    return result
