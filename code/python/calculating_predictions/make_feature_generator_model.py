from __future__ import print_function
import os
import sys
import numpy as np
import pickle

sys.path.append("../utils")
from constants import DAYS_NUMBER
from json_tools import get_linear_stacked_features, get_regression_labels
from pool_iterator import pool_iterator
from log import log


def make_feature_generator_model(args):
    """
    args contain:
        verbose: bool. Wether to print logs to stdout or not.
        data_folder: str. Directory, containing files "day_i.json" where i in range(DAYS_NUMBER).
        out_folder: str. Directory, to save results. 1 file will be created.
        model_constructor: callable. Takes verbose param.
        model_name: str. Created file will have such name.
        train_days: list of ints. Numbers of days to train on.
        need_position_feature: include position in features or not.
        first_feature: int. First feature to take in tarining.
        last_feature: int. Last feature to take in training.
    """
    if args.verbose:
        log("preprocesing started")
    model = args.model_constructor(args.verbose)
    json_filenames = [os.path.join(args.data_folder, "day_{}.json".format(day)) for day in range(DAYS_NUMBER)]

    def get_day_features(day):
        return get_linear_stacked_features(
            pool_iterator(json_filenames[day]),
            first_feature=args.first_feature,
            last_feature=args.last_feature,
            add_positions=args.need_position_feature
        )

    train_features = np.concatenate([get_day_features(day) for day in args.train_days], axis=0)
    if args.verbose:
        log("train features shape: {}".format(np.shape(train_features)))

    def get_day_labels(day):
        return get_regression_labels(pool_iterator(json_filenames[day]))

    train_labels = np.concatenate([get_day_labels(day) for day in args.train_days], axis=0)

    if args.verbose:
        log("preprocesing finished")
        log("start training on days {}".format(args.train_days))
    model.fit(train_features, train_labels)
    if args.verbose:
        log("training finished")
    filename = os.path.join(
        args.out_folder,
        args.model_name + "_trained_on_{}".format('_'.join(map(str, args.train_days)))
    )
    if args.verbose:
        log("saving model in path \"{}\"".format(filename))
    pickle.dump(model, open(filename, "wb"))
