import numpy as np
import sys
sys.path.append('../general')
from metric import metric
sys.path.append('../simple_regression')
from test_regression import get_positions, make_features


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


def get_data_agregation_metrics(pool, model, probas_functions):
    probas_list = [function(pool) for function in probas_functions]

    train_pool, test_pool, train_indecies, test_indecies = pool.train_test_split(need_indicies=True)
    model.fit(make_features(train_pool.features, train_pool.positions), train_pool.targets)
    positions = get_positions(test_pool.features, model)

    metrics = [
        metric(positions, test_pool.positions, test_pool.targets, probas[test_indecies])
        for probas in probas_list
    ]

    return metrics


def test_data_agregation(models, pool, probas_functions):
    scores = [
        get_data_agregation_metrics(pool, model, probas_functions)
        for model in models
    ]
    return scores
