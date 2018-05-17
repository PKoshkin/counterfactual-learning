import argparse
from linear import calculate_linear


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="SVR or logistic")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--first_feature", type=int, required=True)
    parser.add_argument("--last_feature", type=int, required=True)
    args = parser.parse_args()

    calculate_linear(args.model, args.data_folder, args.out_folder, args.first_feature, args.last_feature)


if __name__ == "__main__":
    run()
