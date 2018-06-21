import os
import subprocess
from results_handling import get_results
from strategies import check_existance, QBCMetrics, MixTypes


def _test(strategy, params='{}'):
    assert check_existance(strategy), "Unknown strategy: {}".format(strategy)

    result_file_name = '_test_result_file_{}.txt'.format(strategy)
    test_pool_path = 'test_test_pool.csv'
    train_pool_path = 'test_train_pool.csv'

    process = subprocess.Popen(
        [
            './al_experiment.py',
            '--results', result_file_name,
            '--train_pool', train_pool_path,
            '--val_pool', test_pool_path,
            '--test_pool', test_pool_path,
            '--strategy', strategy,
            '--initial_size', str(0.2),
            '--batch_size', str(0.2),
            '--random_seed', '0',
            '--train_steps', '20',
            '--strategy_params', params,
        ]
    )
    returncode = process.wait()
    assert returncode == 0, "al_experiment failed with error"

    results, _ = get_results(result_file_name)
    # os.remove(result_file_name)
    assert len(results[strategy][0]) == 1
    assert len(results[strategy][0][0]) == 5
    assert len(results[strategy][1]) == 1
    assert (results[strategy][0] == results[strategy][1]).all()


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


def test_PR():
    _test('PR')


def test_QBC():
    _test('QBC')
    _test('QBC', params='{"metric": "' + QBCMetrics.KL + '"}')


def test_mix_EG():
    _test('diversity-density', '{"mix_type": "' + MixTypes.EXPLORATION_GUIDED + '"}')
    _test(
        'diversity-PR',
        '{"mix_type": "' + MixTypes.EXPLORATION_GUIDED + '", "reserve_size": 1.2}'
    )


def test_split_by_positions():
    _test('US', '{"split_by_positions": true}')
    _test('density', '{"split_by_positions": true}')
    _test('QBC', '{"split_by_positions": true}')
    _test('diversity-PR', '{"split_by_positions": true}')
    _test(
        'diversity-PR',
        '{"mix_type": "' + MixTypes.EXPLORATION_GUIDED + '", "split_by_positions": true}'
    )


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


if __name__ == '__main__':
    # test_mix_EG()
    test_split_by_positions()
