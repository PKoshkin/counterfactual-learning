import argparse
import numpy as np
from evaluate import evaluate


class ArgumentException(Exception):
    pass


def argmax_positions_by_predictions(predictions):
    return np.argmax(predictions, axis=-1)


def double_argmax_positions_by_predictions(predictions):
    return np.argmax(np.argmax(predictions, axis=-1), axis=-1)


def main():
    types = ["evaluate_regression", "evaluate_classification"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--predictions_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--type", type=str, required=True, help="str in [{}]".format(", ".join(types)))
    args = parser.parse_args()
    if args.type == "evaluate_regression":
        evaluate(args.predictions_folder, args.data_folder, args.out_folder, argmax_positions_by_predictions)
    elif args.type == "evaluate_classification":
        evaluate(args.predictions_folder, args.data_folder, args.out_folder, double_argmax_positions_by_predictions)
    else:
        raise ArgumentException("Wrong type \"{}\"".format(args.type))


if __name__ == "__main__":
    main()
