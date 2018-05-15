import argparse
from simple_regression import calculate_simple_regression_predictions
import xgboost as xgb
from catboost import CatBoostRegressor


class ArgumentException(Exception):
    pass


def run_regression():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.model.lower() == "catboost":
        model_constructor = lambda: CatBoostRegressor(verbose=args.verbose)
    elif args.model.lower() in ["xgb", "xgboost"]:
        model_constructor = lambda: xgb.XGBRegressor(silent=not args.verbose)
    else:
        raise ArgumentException("Wrong model \"{}\".".format(args.model))

    calculate_simple_regression_predictions(model_constructor, args.data_folder, args.out_folder)

if __name__ == "__main__":
    run_regression()
