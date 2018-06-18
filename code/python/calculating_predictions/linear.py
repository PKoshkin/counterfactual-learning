from calculate_predictions import calculate_predictions

import os
import sys
sys.path.append("../utils")
from constants import FEATURES_NUMBER


def calculate_linear(args):
    out_folder = args.out_folder
    args.type = "binary_regression"
    args.threshold = 0
    args.first_feature = 0
    args.last_feature = args.step
    while args.last_feature < FEATURES_NUMBER:
        args.out_folder = os.path.join(
            out_folder,
            "features_from_{}_to_{}".format(args.first_feature, args.last_feature)
        )
        if not os.path.exists(args.out_folder):
            os.mkdir(args.out_folder)
        calculate_predictions(args)
        args.first_feature += args.step
        args.last_feature += args.step
