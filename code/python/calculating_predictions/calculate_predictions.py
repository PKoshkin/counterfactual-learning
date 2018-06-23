import shutil
import os
from make_predictions import make_predictions
from make_trained_model import make_trained_model
import random


def calculate_predictions(args):
    """
    args contain:
        verbose: bool. Wether to print logs to stdout or not.
        data_folder: str. Directory, containing files "day_i.json" where i in range(DAYS_NUMBER).
        out_folder: str. Directory, to save results. 1 file will be created.
        type: str. One of ["regression", "classification", "binary_classification"]
        position_features_num: int
        model_constructor: callable. Takes verbose param. If type == "classification" also takes max_clicks param.
        args.model_path = (
        model_path: str. path to model
        additional_features: (dict day -> additional features for day) or None.
        add_base_features: bool. Wether to add base features or to use only additional features.
        labels_to_substruct: (dict day -> targets to substruct for day) or None.
        train_days: list of ints. Numbers of days to train on.
        test_days: list of ints. Numbers of days to test on.
        validation_day: int or None. Number of day to validate on.
        first_feature: int. First feature to take in tarining.
        last_feature: int. Last feature to take in training.

        if args.tpye == "classification"
            args.max_clicks should be provided

        if args.type == "binary_classification"
            args.threshold shoul be provided
    """
    folder = "tmp_" + str(random.randint(0, 1e6))
    os.mkdir(folder)
    args.model_path = os.path.join(
        folder,
        "tmp_model_trained_on_{}".format('_'.join(map(str, args.train_days)))
    )
    make_trained_model(args)
    make_predictions(args)
    shutil.rmtree(folder)
