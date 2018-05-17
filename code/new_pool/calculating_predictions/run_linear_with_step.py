import argparse
import sys
import os
from linear import calculate_linear
sys.path.append("../utils")
from constants import FEATURES_NUMBER


def run_with_step():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="SVR or logistic")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    first_feature = 0
    last_feature = args.step
    while first_feature < FEATURES_NUMBER:
        res_dir = os.path.join(args.out_folder, "features_from_{}_to_{}".format(first_feature, last_feature))
        os.mkdir(res_dir)
        calculate_linear(args.model, args.data_folder, res_dir, first_feature, last_feature)
        first_feature += args.step
        last_feature += args.step


if __name__ == "__main__":
    run_with_step()
