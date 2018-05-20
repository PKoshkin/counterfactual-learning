import argparse
from evaluate_regression import evaluate_regression


def main():
    types = ["evaluate_regression"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--predictions_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--type", type=str, required=True, help="str in [{}]".format(", ".join(types)))
    args = parser.parse_args()
    if args.type == "evaluate_regression":
        evaluate_regression(args.predictions_folder, args.data_folder, args.out_folder)


if __name__ == "__main__":
    main()
