import argparse
from evaluate_regression import evaluate_regression
from evaluate_classification import evaluate_classification


class ArgumentException(Exception):
    pass


def main():
    types = ["evaluate_regression", "evaluate_classification"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--predictions_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--type", type=str, required=True, help="str in [{}]".format(", ".join(types)))
    args = parser.parse_args()
    if args.type == "evaluate_regression":
        evaluate_regression(args.predictions_folder, args.data_folder, args.out_folder)
    elif args.type == "evaluate_classification":
        evaluate_classification(args.predictions_folder, args.data_folder, args.out_folder)
    else:
        raise ArgumentException("Wrong type \"{}\"".format(args.type))


if __name__ == "__main__":
    main()
