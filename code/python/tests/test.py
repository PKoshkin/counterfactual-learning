from json import loads as json_from_string
import numpy as np
import sys
import os
from test_utils import new_pool_code_path, pool_filename, days_data_path
from test_utils import make_days_data, clear
sys.path.append("../utils")
from constants import TIMESTAMPS, POSITIONS_VARIANTS, DAYS_NUMBER


def test_pool():
    with open(pool_filename) as handler:
        jsons = [json_from_string(line.strip()) for line in handler]
    assert len(jsons) == DAYS_NUMBER * 2
    for json in jsons:
        assert json["ts"] in TIMESTAMPS


def test_days_data():
    make_days_data()
    assert len(os.listdir(days_data_path)) == DAYS_NUMBER
    for filename in os.listdir(days_data_path):
        filename = os.path.join(days_data_path, filename)
        with open(filename) as handler:
            assert len([line for line in handler]) == 2
    clear([days_data_path])


def train_regression(run_path, days_data_path, res_folder):
    for train_day in range(DAYS_NUMBER - 1):
        os.system("python2 {} --data_folder {} --out_folder {} --model catboost --train_day {} regression".format(
            run_path,
            days_data_path,
            res_folder,
            train_day
        ))


def test_regression():
    make_days_data()
    res_folder = "res"
    os.mkdir(res_folder)
    train_regression(os.path.join(new_pool_code_path, "calculating_predictions/run.py"), days_data_path, res_folder)
    assert len(os.listdir(res_folder)) == DAYS_NUMBER - 1
    for i, filename in enumerate(os.listdir(res_folder)):
        assert "train_{}_test_{}".format(i, i + 1)
        predictions = np.load(os.path.join(res_folder, filename))
        assert len(np.shape(predictions)) == 2
        assert np.shape(predictions)[1] == len(POSITIONS_VARIANTS)
    clear([res_folder, days_data_path])


def train_classification(run_path, days_data_path, res_folder, max_clicks):
    for train_day in range(DAYS_NUMBER - 1):
        os.system("python2 {} --data_folder {} --out_folder {} --model catboost --train_day {} classification --max_clicks {}".format(
            run_path,
            days_data_path,
            res_folder,
            train_day,
            max_clicks
        ))


def test_classification():
    make_days_data()
    res_folder = "res"
    max_clicks = 3
    os.mkdir(res_folder)
    train_classification(
        os.path.join(new_pool_code_path, "calculating_predictions/run.py"),
        days_data_path,
        res_folder,
        max_clicks
    )
    assert len(os.listdir(res_folder)) == DAYS_NUMBER - 1
    for i, filename in enumerate(os.listdir(res_folder)):
        assert "train_{}_test_{}".format(i, i + 1)
        predictions = np.load(os.path.join(res_folder, filename))
        assert len(np.shape(predictions)) == 3
        assert np.shape(predictions)[1:] == (len(POSITIONS_VARIANTS), max_clicks + 2)
    clear([res_folder, days_data_path])


def train_linear(run_path, days_data_path, res_folder, model, step):
    for train_day in range(DAYS_NUMBER - 1):
        os.system("python2 {} --data_folder {} --out_folder {} --model {} --train_day {} linear --step {}".format(
            run_path,
            days_data_path,
            res_folder,
            model,
            train_day,
            step
        ))


def test_linear():
    step = 100
    models = ["svr", "lars", "elastic", "perceptron"]
    res_folder = "res"
    linear_predictions_folders = [
        "{}_linear_predictions".format(model)
        for model in models
    ]
    make_days_data()
    for folder in linear_predictions_folders:
        os.mkdir(folder)
    for model, res_folder in zip(models, linear_predictions_folders):
        train_linear(
            os.path.join(new_pool_code_path, "calculating_predictions/run.py"),
            days_data_path,
            res_folder,
            model,
            step
        )
        assert len(os.listdir(res_folder)) > 0
        for folder in os.listdir(res_folder):
            directory = os.path.join(res_folder, folder)
            assert len(os.listdir(directory)) == DAYS_NUMBER - 1
            for i, filename in enumerate(os.listdir(directory)):
                assert "train_{}_test_{}".format(i, i + 1)
                predictions = np.load(os.path.join(directory, filename))
                assert len(np.shape(predictions)) == 1
    clear([days_data_path] + linear_predictions_folders)


def train_binary_classification(run_path, days_data_path, res_folder, threshold):
    for train_day in range(DAYS_NUMBER - 1):
        os.system("python2 {} --data_folder {} --out_folder {} --model catboost --train_day {} binary_classification --threshold {}".format(
            run_path,
            days_data_path,
            res_folder,
            train_day,
            threshold
        ))


def test_binary_classification():
    make_days_data()
    res_folder = "res"
    threshold = 0
    os.mkdir(res_folder)
    train_binary_classification(
        os.path.join(new_pool_code_path, "calculating_predictions/run.py"),
        days_data_path,
        res_folder,
        threshold
    )
    assert len(os.listdir(res_folder)) == DAYS_NUMBER - 1
    for i, filename in enumerate(os.listdir(res_folder)):
        assert "train_{}_test_{}".format(i, i + 1)
        predictions = np.load(os.path.join(res_folder, filename))
        assert len(np.shape(predictions)) == 2
        assert np.shape(predictions)[1] == 2
    clear([res_folder, days_data_path])


def train_linear_stacking(run_path, days_data_path, res_folder, linear_predictions_folders, max_clicks):
    for train_day in range(1, DAYS_NUMBER - 1):
        os.system("python2 {} --data_folder {} --out_folder {} --model catboost --train_day {} linear_stacking --linear_predictions {} --max_clicks {}".format(
            run_path,
            days_data_path,
            res_folder,
            train_day,
            ' '.join(linear_predictions_folders),
            max_clicks
        ))


def test_linear_stacking():
    max_clicks = 3
    step = 100
    models = ["svr", "lars", "elastic", "perceptron"]
    res_folder = "res"
    linear_predictions_folders = [
        "{}_linear_predictions".format(model)
        for model in models
    ]
    make_days_data()
    os.mkdir(res_folder)
    for folder in linear_predictions_folders:
        os.mkdir(folder)
    for model, predictions_folder in zip(models, linear_predictions_folders):
        train_linear(
            os.path.join(new_pool_code_path, "calculating_predictions/run.py"),
            days_data_path,
            predictions_folder,
            model,
            step
        )
    train_linear_stacking(
        os.path.join(new_pool_code_path, "calculating_predictions/run.py"),
        days_data_path,
        res_folder,
        linear_predictions_folders,
        max_clicks
    )

    assert len(os.listdir(res_folder)) == DAYS_NUMBER - 2
    for i, filename in enumerate(os.listdir(res_folder)):
        assert "train_{}_test_{}".format(i + 1, i + 2)
        predictions = np.load(os.path.join(res_folder, filename))
        assert len(np.shape(predictions)) == 3
        assert np.shape(predictions)[1:] == (len(POSITIONS_VARIANTS), max_clicks + 2)
    clear([res_folder, days_data_path] + linear_predictions_folders)


def test_regression_evaluation():
    make_days_data()
    res_folder = "res"
    metrics_folder = "metrics"
    os.mkdir(res_folder)
    os.mkdir(metrics_folder)
    train_regression(os.path.join(new_pool_code_path, "calculating_predictions/run.py"), days_data_path, res_folder)

    os.system("python2 {} --data_folder {} --predictions_folder {} --out_folder {} --type argmax_regression".format(
        os.path.join(new_pool_code_path, "evaluation/run_evaluation.py"),
        days_data_path,
        res_folder,
        metrics_folder
    ))
    assert len([line for line in open(os.path.join(metrics_folder, "metrics.txt"))]) == DAYS_NUMBER - 1
    clear([res_folder, days_data_path, metrics_folder])


def test_classification_evaluation():
    max_clicks = 3
    res_folder = "res"
    metrics_folder = "metrics"
    make_days_data()
    os.mkdir(res_folder)
    os.mkdir(metrics_folder)
    train_classification(
        os.path.join(new_pool_code_path, "calculating_predictions/run.py"),
        days_data_path,
        res_folder,
        max_clicks
    )

    os.system("python2 {} --data_folder {} --predictions_folder {} --out_folder {} --type argmax_classification".format(
        os.path.join(new_pool_code_path, "evaluation/run_evaluation.py"),
        days_data_path,
        res_folder,
        metrics_folder
    ))
    assert len([line for line in open(os.path.join(metrics_folder, "metrics.txt"))]) == DAYS_NUMBER - 1
    clear([res_folder, days_data_path, metrics_folder])


def test_linear_stacking_evaluation():
    max_clicks = 3
    step = 100
    models = ["svr", "lars", "elastic", "perceptron"]
    res_folder = "res"
    metrics_folder = "metrics"
    linear_predictions_folders = [
        "{}_linear_predictions".format(model)
        for model in models
    ]
    make_days_data()
    os.mkdir(res_folder)
    os.mkdir(metrics_folder)
    for folder in linear_predictions_folders:
        os.mkdir(folder)
    for model, predictions_folder in zip(models, linear_predictions_folders):
        train_linear(
            os.path.join(new_pool_code_path, "calculating_predictions/run.py"),
            days_data_path,
            predictions_folder,
            model,
            step
        )
    train_linear_stacking(
        os.path.join(new_pool_code_path, "calculating_predictions/run.py"),
        days_data_path,
        res_folder,
        linear_predictions_folders,
        max_clicks
    )
    os.system("python2 {} --data_folder {} --predictions_folder {} --out_folder {} --type argmax_classification".format(
        os.path.join(new_pool_code_path, "evaluation/run_evaluation.py"),
        days_data_path,
        res_folder,
        metrics_folder
    ))
    assert len([line for line in open(os.path.join(metrics_folder, "metrics.txt"))]) == DAYS_NUMBER - 2
    clear([res_folder, days_data_path, metrics_folder] + linear_predictions_folders)
