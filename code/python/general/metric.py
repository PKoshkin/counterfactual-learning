import numpy as np


def metric(answers_positions, target_positions, target, target_probs):
    mask = target_probs != 0
    return np.mean(
        target[mask] / target_probs[mask] * (answers_positions[mask] == target_positions[mask])
    )


def normed_metric(answers_positions, target_positions, target, target_probs):
    return np.sum(
        target / target_probs * (answers_positions == target_positions)
    ) / np.sum(
        1 / target_probs * (answers_positions == target_positions)
    )


def get_metric(pool, positions):
    return metric(positions, pool.positions, pool.targets, pool.probas)


def get_benefit(pool, weights=None):
    if weights is None:
        return np.mean(pool.targets)
    else:
        if len(weights) != len(pool.features):
            raise Exception("Wrong weights len")
        return np.mean(pool.targets * weights) / np.mean(weights)


def prod_metric(pool):
    scores = []
    for i in range(10):
        _, test_pool = pool.train_test_split()
        scores.append(metric(
            test_pool.prod_positions,
            test_pool.positions,
            test_pool.targets,
            test_pool.probas
        ))
    return scores


def max_metric(pool):
    scores = []
    for i in range(10):
        _, test_pool = pool.train_test_split()
        mask = test_pool.targets > 0
        scores.append(metric(
            test_pool.positions[mask],
            test_pool.positions[mask],
            test_pool.targets[mask],
            test_pool.probas[mask]
        ))
    return scores
