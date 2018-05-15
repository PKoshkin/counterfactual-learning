import argparse
import xgboost as xgb
from train_and_test_simple_regression import train_and_test_simple_regression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument('--verbose', action='store_false')
    args = parser.parse_args()
    train_and_test_simple_regression(
        lambda: xgb.XGBRegressor(silent=args.verbose),
        args.data_folder,
        args.out_folder
    )


if __name__ == "__main__":
    main()
