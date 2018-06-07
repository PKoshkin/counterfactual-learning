from json import loads as json_from_string
import sys
import os
sys.path.append("../utils")
from constants import DAYS_NUMBER, TIMESTAMPS


data_path = "../../../sorted_data"
new_pool_code_path = "../"
pool_filename = os.path.join(data_path, "pool.json")


def test_pool():
    with open(pool_filename) as handler:
        jsons = [json_from_string(line.strip()) for line in handler]
    assert len(jsons) == DAYS_NUMBER * 2
    for json in jsons:
        assert json["ts"] in TIMESTAMPS


def test_days_data():
    days_data_path = os.path.join(data_path, "days_data")
    os.mkdir(days_data_path)
    os.system("python2 {} --pool_path {} --skip_prob 0.001 --out_folder {}".format(
        os.path.join(new_pool_code_path, "utils/make_days_jsons.py"),
        pool_filename,
        days_data_path
    ))
    assert len(os.listdir(days_data_path)) == DAYS_NUMBER
    for filename in os.listdir(days_data_path):
        filename = os.path.join(days_data_path, filename)
        with open(filename) as handler:
            assert len([line for line in handler]) == 2
        os.remove(filename)
    os.rmdir(days_data_path)
