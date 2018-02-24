import numpy as np
import sys
sys.path.append('../general')
from metric import metric
from constants import POSITION_VARIANTS


def make_features(main_features, positions):
    return np.concatenate((main_features, np.reshape(positions, (-1, 1))), axis=1)


def get_positions(features_to_answer, model):
    positions = []
    for features in features_to_answer:
        features = np.repeat(np.reshape(features, (1, -1)), len(POSITION_VARIANTS), axis=0)
        predicted_scores = model.predict(make_features(features, POSITION_VARIANTS))
        positions.append(np.argmax(predicted_scores))

        # mask = np.any(predicted_scores > 0, axis=1)
        # show_positions = np.argmax(predicted_scores, axis=1)
        # positions = show_positions * mask + NONE_POSITION * (1 - mask)
    return positions


def get_metric(pool, model):
    train_pool, test_pool = pool.train_test_split()
    model.fit(make_features(train_pool.features, train_pool.positions), train_pool.targets)
    positions = get_positions(test_pool.features, model)

    return metric(positions, test_pool.positions, test_pool.targets, test_pool.probas)


def test_regression(models, pool):
    scores = [
        get_metric(pool, model)
        for model in models
    ]
    return scores
