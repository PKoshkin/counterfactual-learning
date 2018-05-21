import argparse
import numpy as np
from evaluate import evaluate


class ArgumentException(Exception):
    pass


def argmax_positions(predictions):
    return np.argmax(predictions, axis=-1)


def double_argmax_positions(predictions):
    return np.argmax(np.argmax(predictions, axis=-1), axis=-1)


def weighted_positions(predictions):
    repeated_arange = np.repeat([np.arange(np.shape(predictions)[1])], np.shape(predictions)[0], axis=0)
    return np.round(np.sum(predictions * repeated_arange, axis=1) / np.sum(predictions, axis=1))


def expect_weighted_positions(predictions):
    repeated_arange = np.repeat(
        [np.repeat([np.arange(np.shape(predictions)[2])], np.shape(predictions)[1], axis=0)],
        np.shape(predictions)[0],
        axis=0
    )
    like_regression_predictions = np.sum(repeated_arange * predictions, axis=-1)
    return weighted_positions(like_regression_predictions)


def expect_argmax_positions(predictions):
    repeated_arange = np.repeat(
        [np.repeat([np.arange(np.shape(predictions)[2])], np.shape(predictions)[1], axis=0)],
        np.shape(predictions)[0],
        axis=0
    )
    like_regression_predictions = np.sum(repeated_arange * predictions, axis=-1)
    return argmax_positions(like_regression_predictions)


def argmax_weighted_positions(predictions):
    like_regression_predictions = np.argmax(predictions, axis=-1)
    return weighted_positions(like_regression_predictions)


def main():
    types = ["argmax_regression",
             "weighted_regression",
             "argmax_classification",
             "expect_weighted_classification",
             "argmax_weighted_classification",
             "expect_argmax_classification"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--predictions_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--type", type=str, required=True, help="str in [{}]".format(", ".join(types)))
    args = parser.parse_args()
    if args.type == "argmax_regression":
        evaluate(args.predictions_folder, args.data_folder, args.out_folder, argmax_positions)
    elif args.type == "argmax_classification":
        evaluate(args.predictions_folder, args.data_folder, args.out_folder, double_argmax_positions)
    elif args.type == "weighted_regression":
        evaluate(args.predictions_folder, args.data_folder, args.out_folder, weighted_positions)
    elif args.type == "expect_weighted_classification":
        evaluate(args.predictions_folder, args.data_folder, args.out_folder, expect_weighted_positions)
    elif args.type == "argmax_weighted_classification":
        evaluate(args.predictions_folder, args.data_folder, args.out_folder, argmax_weighted_positions)
    elif args.type == "expect_argmax_classification":
        evaluate(args.predictions_folder, args.data_folder, args.out_folder, expect_argmax_positions)
    else:
        raise ArgumentException("Wrong type \"{}\"".format(args.type))


if __name__ == "__main__":
    main()
