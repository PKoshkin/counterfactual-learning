from sklearn.svm import LinearSVR
from sklearn.linear_model import Lars, ElasticNet, Perceptron

from regression import calculate_regression_predictions

import sys
sys.path.append("../utils")
from constants import FEATURES_NUMBER




class ArgumentException(Exception):
    pass


def calculate_linear(args):
    if args.model.lower() == "svr":
        args.model_constructor = LinearSVR
    elif args.model.lower() == "lars":
        args.model_constructor = Lars
    elif args.model.lower() == "elastic":
        args.model_constructor = ElasticNet
    elif args.model.lower() == "perceptron":
        args.model_constructor = Perceptron
    else:
        raise ArgumentException("Wrong model \"{}\".".format(args.model))

    out_folder = args.out_folder

    args.first_feature = 0
    args.last_feature = args.step
    while args.first_feature < FEATURES_NUMBER:
        args.out_folder= os.path.join(out_folder, "features_from_{}_to_{}".format(args.first_feature, args.last_feature))
        os.mkdir(args.out_folder)
        calculate_regression_predictions(args)
        args.first_feature += args.step
        args.last_feature += args.step

