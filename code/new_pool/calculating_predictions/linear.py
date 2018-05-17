from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression

from regression import calculate_regression_predictions


class ArgumentException(Exception):
    pass


def calculate_linear(model, data_folder, out_folder, first_feature, last_feature):
    if model.lower() == "svr":
        model_constructor = LinearSVR
    elif model.lower() == "logistic":
        model_constructor = LogisticRegression
    else:
        raise ArgumentException("Wrong model \"{}\".".format(model))
    calculate_regression_predictions(model_constructor,
                                     data_folder,
                                     out_folder,
                                     first_feature,
                                     last_feature)
