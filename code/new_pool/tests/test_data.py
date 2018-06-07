from json import loads as json_from_string
import sys
sys.path.append("../utils")
from constants import DAYS_NUMBER, TIMESTAMPS


def test_pool():
    pool_path = "../../../sorted_data/pool.json"
    with open(pool_path) as handler:
        jsons = [json_from_string(line.strip()) for line in handler]
    assert len(jsons) == DAYS_NUMBER * 2
    for json in jsons:
        assert json["ts"] in TIMESTAMPS
