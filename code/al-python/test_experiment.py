import os
import subprocess
from results_handling import parse_result_file
from strategies import check_existance, QBCMetrics


def _test(strategy, params='{}'):
    assert check_existance(strategy), "Unknown strategy: {}".format(strategy)

    result_file_name = '_test_result_file.txt'
    test_pool_path = 'test_test_pool.csv'
    train_pool_path = 'test_train_pool.csv'

    process = subprocess.Popen(
        [
            './al_experiment.py',
            '--results', result_file_name,
            '--train_pool', train_pool_path,
            '--test_pool', test_pool_path,
            '--strategy', strategy,
            '--initial_size', '5',
            '--batch_size', '20',
            '--random_seed', '0',
            '--train_steps', '20',
            '--strategy_params', params,
        ]
    )
    returncode = process.wait()
    assert returncode == 0, "al_experiment failed with error"

    _, results = parse_result_file(result_file_name)
    os.remove(result_file_name)
    assert len(results) == 4


def test_US():
    _test('US')


def test_random():
    _test('random')


def test_density():
    _test('density')


def test_diversity():
    _test('diversity')


def test_mix():
    _test('US-density')
    _test('density-diversity')


def test_density_params():
    _test('density', '{"share": 0.1}')


def test_diversity_params():
    _test('diversity', '{"share": 0.1}')


def test_US_params():
    _test('US', '{"uncertainty_metric": "gini"}')
    _test('US', '{"uncertainty_metric": "gini", "proba_normalization": "softmax"}')
    _test('US', '{"uncertainty_metric": "gini", "proba_normalization": "softmax", "T": 0.01}')
    _test('US', '{"uncertainty_metric": "entropy"}')
    _test('US', '{"uncertainty_metric": "entropy", "proba_normalization": "softmax"}')
    _test('US', '{"uncertainty_metric": "entropy", "proba_normalization": "softmax", "T": 0.01}')


def test_mix_params():
    _test('US-density', '{"US": {"uncertainty_metric": "gini"}, "density": {"share": 0.1}}')


def test_PR():
    _test('PR')


def test_QBC():
    _test('QBC')
    _test('QBC', params='{"metric": "' + QBCMetrics.KL + '"}')


if __name__ == '__main__':
    test_US_params()
