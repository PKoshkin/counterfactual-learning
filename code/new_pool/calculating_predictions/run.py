import argparse
import os
import sys
from regression import calculate_regression_predictions
from simple_classification import calculate_simple_classification_predictions
from binary_classification import calculate_binary_classification_predictions
from linear_stacking import calculate_classification_stacked_on_linear_predictions
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
        raise ArgumentException("\"max_clicks\" argument is valid only for regression")
    if args.threshold is not None:
        raise ArgumentException("\"threshold\" argument is not valid for binary regression")
    if args.linear_predictions is not None:
        raise ArgumentException("\"linear_predictions\" argument is not valid for regression")
    if args.model.lower() == "catboost":
        model_constructor = lambda: CatBoostRegressor(verbose=args.verbose)
    elif args.model.lower() in ["xgb", "xgboost"]:
        model_constructor = lambda: xgb.XGBRegressor(silent=not args.verbose)
    else:
        raise ArgumentException("Wrong model \"{}\".".format(args.model))
    calculate_regression_predictions(model_constructor, args.data_folder, args.out_folder)


def calculate_binary_classification(args):
    if args.model != "catboost":
        raise ArgumentException("Only catboost is supported now!")
    if args.threshold is None:
        raise ArgumentException("\"threshold\" argument is required for binary classification")
    if args.max_clicks is not None:
        raise ArgumentException("\"max_clicks\" argument is valid only for binary classification")
    if args.linear_predictions is not None:
        raise ArgumentException("\"linear_predictions\" argument is not valid for binary classification")
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
        raise ArgumentException("\"threshold\" argument is not valid for classification")
    if args.linear_predictions is not None:
        raise ArgumentException("\"linear_predictions\" argument is not valid for classification")
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


def calculate_classification_stacked_on_linear(args):
    if args.model != "catboost":
        raise ArgumentException("Only catboost is supported now!")
    if args.max_clicks is None:
        raise ArgumentException("\"max_clicks\" argument is required for linear_stacking")
    if args.linear_predictions is None:
        raise ArgumentException("\"linear_predictions\" argument is required for linear_stacking")
    if args.threshold is not None:
        raise ArgumentException("\"threshold\" argument is not valid for linear_stacking")
    if args.model.lower() == "catboost":
        model_constructor = lambda: CatBoostClassifier(verbose=args.verbose,
                                                       loss_function='MultiClass',
                                                       classes_count=args.max_clicks + 2)
    elif args.model.lower() in ["xgb", "xgboost"]:
        model_constructor = lambda: xgb.XGBClassifier(silent=not args.verbose)
    else:
        raise ArgumentException("Wrong model \"{}\".".format(args.model))
    calculate_classification_stacked_on_linear_predictions(model_constructor,
                                                           args.data_folder,
                                                           args.out_folder,
                                                           args.max_clicks,
                                                           args.linear_predictions)


def calculate_linear_with_step(args):
    if args.max_clicks is not None:
        raise ArgumentException("\"max_clicks\" argument is valid only for linear")
    if args.threshold is not None:
        raise ArgumentException("\"threshold\" argument is not valid for linear")
    if args.linear_predictions is not None:
        raise ArgumentException("\"linear_predictions\" argument is not valid for linear")
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
    types = ["classification", "regression", "binary_classification", "linear", "linear_stacking"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="catboost of xgboost")
    parser.add_argument("--type", type=str, required=True, help="str in [{}]".format(", ".join(types)))
    parser.add_argument("--linear_predictions", type=str, nargs='*', help="list of folders with linear regression predictions")
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
    elif args.type == "linear_stacking":
        calculate_classification_stacked_on_linear(args)
    else:
        raise ArgumentException("Wrong model type \"{}\".".format(args.type))


if __name__ == "__main__":
    run()
