import argparse
import os
import sys
from simple_regression import calculate_simple_regression_predictions
from simple_classification import calculate_simple_classification_predictions
from binary_classification import calculate_binary_classification_predictions
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from linear import calculate_linear
sys.path.append("../utils")
from constants import FEATURES_NUMBER


class ArgumentException(Exception):
    pass


def calculate_regression(args):
    if args.model != "catboost":
        raise ArgumentException("Only catboost is supported now!")
    if args.max_clicks is not None:
        raise ArgumentException("\"max_clicks\" argument is valid only for classification")
    if args.threshold is not None:
        raise ArgumentException("\"threshold\" argument is not valid for binary classification")
    if args.model.lower() == "catboost":
        model_constructor = lambda: CatBoostRegressor(verbose=args.verbose)
    elif args.model.lower() in ["xgb", "xgboost"]:
        model_constructor = lambda: xgb.XGBRegressor(silent=not args.verbose)
    else:
        raise ArgumentException("Wrong model \"{}\".".format(args.model))
    calculate_simple_regression_predictions(model_constructor, args.data_folder, args.out_folder)


def calculate_binary_classification(args):
    if args.model != "catboost":
        raise ArgumentException("Only catboost is supported now!")
    if args.threshold is None:
        raise ArgumentException("\"threshold\" argument is required for binary classification")
    if args.max_clicks is not None:
        raise ArgumentException("\"max_clicks\" argument is valid only for classification")
    if args.model.lower() == "catboost":
        model_constructor = lambda: CatBoostClassifier(verbose=args.verbose)
    elif args.model.lower() in ["xgb", "xgboost"]:
        model_constructor = lambda: xgb.XGBClassifier(silent=not args.verbose)
    else:
        raise ArgumentException("Wrong model \"{}\".".format(args.model))
    calculate_binary_classification_predictions(model_constructor,
                                                args.data_folder,
                                                args.out_folder,
                                                args.threshold)


def calculate_classification(args):
    if args.model != "catboost":
        raise ArgumentException("Only catboost is supported now!")
    if args.max_clicks is None:
        raise ArgumentException("\"max_clicks\" argument is required for classification")
    if args.threshold is not None:
        raise ArgumentException("\"threshold\" argument is not valid for binary classification")
    if args.model.lower() == "catboost":
        model_constructor = lambda: CatBoostClassifier(verbose=args.verbose,
                                                       loss_function='MultiClass',
                                                       classes_count=args.max_clicks + 2)
    elif args.model.lower() in ["xgb", "xgboost"]:
        model_constructor = lambda: xgb.XGBClassifier(silent=not args.verbose)
    else:
        raise ArgumentException("Wrong model \"{}\".".format(args.model))
    calculate_simple_classification_predictions(model_constructor,
                                                args.data_folder,
                                                args.out_folder,
                                                args.max_clicks)


def calculate_linear_with_step(args):
    if args.max_clicks is not None:
        raise ArgumentException("\"max_clicks\" argument is valid only for linear")
    if args.threshold is not None:
        raise ArgumentException("\"threshold\" argument is not valid for linear")
    if args.step is None:
        raise ArgumentException("\"step\" argument is required for lenear")
    first_feature = 0
    last_feature = args.step
    while first_feature < FEATURES_NUMBER:
        res_dir = os.path.join(args.out_folder, "features_from_{}_to_{}".format(first_feature, last_feature))
        os.mkdir(res_dir)
        calculate_linear(args.model, args.data_folder, res_dir, first_feature, last_feature)
        first_feature += args.step
        last_feature += args.step


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="catboost of xgboost")
    parser.add_argument("--type", type=str, required=True, help="classification of regression")
    parser.add_argument("--max_clicks", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--step", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    if not args.fast:
        raise ArgumentException("Only fast mode is supported now!")

    if args.type == "regression":
        calculate_regression(args)
    elif args.type == "classification":
        calculate_classification(args)
    elif args.type == "binary_classification":
        calculate_binary_classification(args)
    elif args.type == "linear":
        calculate_linear_with_step(args)
    else:
        raise ArgumentException("Wrong model type \"{}\".".format(args.type))


if __name__ == "__main__":
    run()
