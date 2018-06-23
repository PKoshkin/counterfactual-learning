from __future__ import print_function
import os
import sys
import numpy as np
from catboost import Pool
import pickle

sys.path.append("../utils")
from constants import DAYS_NUMBER
from json_tools import get_day_features, get_day_labels
from log import log


def make_trained_model(args):
    """
    args contain:
        verbose: bool. Wether to print logs to stdout or not.
        data_folder: str. Directory, containing files "day_i.json" where i in range(DAYS_NUMBER).
        out_folder: str. Directory, to save results. 1 file will be created.
        type: str. One of ["regression", "classification", "binary_classification"]
        position_features_num: int
        model_constructor: callable. Takes verbose param. If type == "classification" also takes max_clicks param.
        model_path: str. path to model
        additional_features: (dict day -> additional features for day) or None.
        add_base_features: bool. Wether to add base features or to use only additional features.
        labels_to_substruct: (dict day -> targets to substruct for day) or None.
        train_days: list of ints. Numbers of days to train on.
        validation_day: int or None. Number of day to validate on.
        first_feature: int. First feature to take in tarining.
        last_feature: int. Last feature to take in training.

        if args.tpye == "classification"
            args.max_clicks should be provided

        if args.type == "binary_classification"
            args.threshold shoul be provided
    """
    assert args.type in ["classification", "regression", "binary_classification"]

    if args.verbose:
        log("preprocesing started")

    if args.type == "classification":
        model = args.model_constructor(args.verbose, args.max_clicks)
    else:
        model = args.model_constructor(args.verbose)

    json_filenames = [os.path.join(args.data_folder, "day_{}.json".format(day)) for day in range(DAYS_NUMBER)]

    train_features = np.concatenate(
        [get_day_features(json_filenames[day], day, args) for day in args.train_days],
        axis=0
    )
    if args.verbose:
        log("train features shape: {}".format(np.shape(train_features)))

    train_labels = np.concatenate([get_day_labels(json_filenames[day], day, args) for day in args.train_days], axis=0)

    if args.verbose:
        log("preprocesing finished")
        log("start training on days {}".format(args.train_days))

    if args.validation_day is not None:
        validation_pool = Pool(
            get_day_features(json_filenames[args.validation_day], args.validation_day, args),
            get_day_labels(json_filenames[args.validation_day], args.validation_day, args)
        )
        if args.verbose:
            log("using fit with validation")
        model.set_params(iterations=2000)
        model.fit(
            train_features,
            train_labels,
            eval_set=validation_pool,
            use_best_model=True
        )
    else:
        if args.verbose:
            log("using fit without validation")
        model.fit(train_features, train_labels)

    if "tree_count_" in dir(model):
        if args.verbose:
            log("built {} trees".format(model.tree_count_))

    if args.verbose:
        log("saving model in path \"{}\"".format(args.model_path))
    pickle.dump(model, open(args.model_path, "wb"))
