from json import loads as json_from_string
import numpy as np
import sys
import os
from test_utils import new_pool_code_path, pool_filename, days_data_path
from test_utils import make_days_data, clear
sys.path.append("../utils")
from constants import DAYS_NUMBER, TIMESTAMPS, POSITIONS_VARIANTS


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
    clear()


def test_regression():
    make_days_data()
    os.mkdir("res")
    os.system("python2 {} --data_folder {} --out_folder res --model catboost --type regression --fast --verbose".format(
        os.path.join(new_pool_code_path, "calculating_predictions/run.py"),
        days_data_path
    ))
    # (DAYS_NUMBER - 1) result file and one times.txt file
    assert len(os.listdir("res")) == DAYS_NUMBER

    for filename in os.listdir("res"):
        if filename != "times.txt":
            predictions = np.load(os.path.join("res", filename))
            assert len(np.shape(predictions)) == 2
            assert np.shape(predictions)[1] == len(POSITIONS_VARIANTS)

    clear()


def test_classification():
    max_clicks = 3
    make_days_data()
    os.mkdir("res")
    os.system("python2 {} --data_folder {} --out_folder res --model catboost --type classification --max_clicks {} --fast --verbose".format(
        os.path.join(new_pool_code_path, "calculating_predictions/run.py"),
        days_data_path,
        max_clicks
    ))
    # (DAYS_NUMBER - 1) result file and one times.txt file
    assert len(os.listdir("res")) == DAYS_NUMBER

    for filename in os.listdir("res"):
        if filename != "times.txt":
            predictions = np.load(os.path.join("res", filename))
            assert len(np.shape(predictions)) == 3
            assert np.shape(predictions)[1:] == (len(POSITIONS_VARIANTS), max_clicks + 2)

    clear()


def test_linear():
    step = 100
    models = ["svr", "lars", "elastic", "perceptron"]
    for model in models:
        make_days_data()
        os.mkdir("res")
        os.system("python2 {} --data_folder {} --out_folder res --model {} --type linear --fast --step {} --verbose".format(
            os.path.join(new_pool_code_path, "calculating_predictions/run.py"),
            days_data_path,
            model,
            step
        ))
        # (DAYS_NUMBER - 1) result file and one times.txt file
        for folder in os.listdir("res"):
            directory = os.path.join("res", folder)
            assert len(os.listdir(directory)) == DAYS_NUMBER

            for filename in os.listdir(directory):
                if filename != "times.txt":
                    predictions = np.load(os.path.join(directory, filename))
                    assert len(np.shape(predictions)) == 2
                    assert np.shape(predictions)[1] == len(POSITIONS_VARIANTS)

        clear()
