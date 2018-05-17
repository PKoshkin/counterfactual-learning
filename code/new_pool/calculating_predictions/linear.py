from sklearn.svm import LinearSVR
from sklearn.linear_model import Lars, ARDRegression, ElasticNet, Perceptron


from regression import calculate_regression_predictions


class ArgumentException(Exception):
    pass


def calculate_linear(model, data_folder, out_folder, first_feature, last_feature):
    if model.lower() == "svr":
        model_constructor = LinearSVR
    elif model.lower() == "lars":
        model_constructor = Lars
    elif model.lower() == "ard":
        model_constructor = ARDRegression
    elif model.lower() == "elastic":
        model_constructor = ElasticNet
    elif model.lower() == "perceptron":
        model_constructor = Perceptron
    else:
        raise ArgumentException("Wrong model \"{}\".".format(model))
    calculate_regression_predictions(model_constructor,
                                     data_folder,
                                     out_folder,
                                     first_feature,
                                     last_feature)
