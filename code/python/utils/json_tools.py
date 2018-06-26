import numpy as np
import pickle
from datetime import datetime
from log import log
from constants import POSITIONS_VARIANTS, AVERAGE_CONSTANTS_METRICS, FEATURES_NUMBER
from pool_iterator import pool_iterator


def get_from_pool(pool_iterator, name, constructor=float):
    return np.array([constructor(item[name]) for item in pool_iterator])


def get_binary_labels(pool_iterator, threshold):
    return get_from_pool(pool_iterator, "target", lambda item: 1 if item > threshold else 0)


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


def make_feature(json,
                 position_features_num,
                 first_feature,
                 last_feature,
                 different_positions,
                 add_ts_features=False):
    assert position_features_num in [0, 1, 2]

    factors = json["factors"][first_feature:last_feature]
    if add_ts_features:
        factors += [
            datetime.fromtimestamp(json["ts"]).day / 32,
            datetime.fromtimestamp(json["ts"]).hour / 24,
            datetime.fromtimestamp(json["ts"]).minute / 60,
            datetime.fromtimestamp(json["ts"]).month / 12
        ]

    def position_features(position):
        if position_features_num == 1:
            return [position]
        elif position_features_num == 2:
            return [position, AVERAGE_CONSTANTS_METRICS[position]]
        else:
            return []

    if different_positions:
        assert position_features_num > 0
        return [position_features(position) + factors for position in POSITIONS_VARIANTS]
    else:
        return position_features(json["pos"]) + factors


def get_linear_stacked_features(pool_iterator,
                                models_list=[],
                                first_feature=0,
                                last_feature=FEATURES_NUMBER,
                                position_features_num=1,
                                different_positions=False,
                                add_base_features=True,
                                verbose=False):
    """
    models_list: list of strings - filenames of files with models to predict features
    """
    assert first_feature >= 0
    assert first_feature < last_feature
    assert last_feature <= FEATURES_NUMBER
    if not add_base_features:
        assert len(models_list) > 0

    base_features = np.array([
        make_feature(item, position_features_num, first_feature, last_feature, different_positions)
        for item in pool_iterator
    ])
    if different_positions:
        base_features = np.reshape(base_features, [-1, np.shape(base_features)[2]])
    if verbose:
        log("    base features shape: {}".format(np.shape(base_features)))
    predictions = []
    for filename in models_list:
        model = pickle.load(open(filename, "rb"))
        # filename has template:
        #    model_all_features_trained_on_<days>
        #    or
        #    model_features_from_<x>_to_<y>_trained_on_<days>
        if filename.find("from") == -1:
            features_to_predict = base_features
        else:
            from_index = filename.find("_from_")
            trained_index = filename.find("_trained_")
            to_index = filename.find("_to_")
            low = int(filename[(from_index + 6):to_index])
            hight = int(filename[(to_index + 4):trained_index])
            position_features = base_features[:, 0:position_features_num]
            factors = base_features[:, (low + position_features_num):(hight + position_features_num)]
            features_to_predict = np.concatenate([position_features, factors], axis=1)
        prediction = model.predict(features_to_predict)
        predictions.append(np.reshape(prediction, [-1, 1]))

    if add_base_features:
        result = np.concatenate([base_features] + predictions, axis=1)
    else:
        result = np.concatenate(predictions, axis=1)
    if verbose:
        log("    {} features added".format(len(predictions)))
        if add_base_features:
            log("    base features included")
        else:
            log("    base features not included")
        log("    result shape: {}".format(np.shape(result)))
    return result


def get_day_features(json_filename, day, args, different_positions=False):
    if args.additional_features is not None:
        return get_linear_stacked_features(
            pool_iterator(json_filename),
            args.additional_features[day],
            first_feature=args.first_feature,
            last_feature=args.last_feature,
            position_features_num=args.position_features_num,
            different_positions=different_positions,
            add_base_features=args.add_base_features,
            verbose=args.verbose
        )
    else:
        return get_linear_stacked_features(
            pool_iterator(json_filename),
            first_feature=args.first_feature,
            last_feature=args.last_feature,
            position_features_num=args.position_features_num,
            different_positions=different_positions,
            add_base_features=args.add_base_features,
            verbose=args.verbose
        )


def get_day_labels(json_filename, day, args):
    if args.labels_to_substruct is not None:
        labels_to_substruct = np.load(args.labels_to_substruct[day])
        day_labels = get_labels(pool_iterator(json_filename), args)
        assert np.shape(labels_to_substruct) == np.shape(day_labels)
        return day_labels - labels_to_substruct
    else:
        return get_labels(pool_iterator(json_filename), args)
