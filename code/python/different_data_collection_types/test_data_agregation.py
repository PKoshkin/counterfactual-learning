import numpy as np
import sys
sys.path.append('../general')
from metric import metric
sys.path.append('../simple_regression')
from test_regression import get_positions, make_features


class TestDataAgregationException(Exception):
    pass


def make_uniform_probas(pool):
    unique_queries = list(set(map(tuple, pool.queries)))
    positions_for_queries = {
        query: []
        for query in unique_queries
    }
    for position, query in zip(pool.positions, pool.queries):
        positions_for_queries[tuple(query)].append(position)
    return np.array([
        1 / len(positions_for_queries[tuple(query)])
        for query in pool.queries
    ])


def make_uniform_near_prod_probas(pool, d=1):
    copy = pool.copy()
    mask = (np.abs(pool.positions - pool.prod_positions) <= d).astype(int)
    copy.filter(mask)
    uniform_features = make_uniform_probas(copy)
    uniform_indecies = np.cumsum(mask)
    uniform_indecies[mask == 0] = 0
    result_features = np.zeros(len(pool.features))
    result_features[uniform_indecies] = uniform_features
    return result_features


def get_data_agregation_metrics(pool, models_list, probas_functions):
    train_pool, test_pool = pool.train_test_split()
    probas_list = [function(train_pool) for function in probas_functions]
    train_pools = [
        train_pool.sample_pool(probas, len(train_pool.features))
        for probas in probas_list
    ]

    positions_list = []
    for model, train_pool in zip(models_list, train_pools):
        model.fit(make_features(train_pool.features, train_pool.positions), train_pool.targets)
        positions_list.append(get_positions(test_pool.features, model))

    metrics = [
        metric(positions, test_pool.positions, test_pool.targets, test_pool.probas)
        for positions in positions_list
    ]

    return metrics


def test_data_agregation(pool, models_constructors, probas_functions, runs=5):
    if len(models_constructors) != len(probas_functions):
        raise TestDataAgregationException()
    scores = [
        get_data_agregation_metrics(
            pool,
            [models_constructor() for models_constructor in models_constructors],
            probas_functions
        ) for i in range(runs)
    ]
    return np.array(scores).T
