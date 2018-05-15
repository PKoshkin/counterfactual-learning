import argparse
from train_and_test_simple_regression import train_and_test_simple_regression


def run(model_constructor):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    train_and_test_simple_regression(lambda: model_constructor(args.verbose), args.data_folder, args.out_folder)
