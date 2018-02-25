import numpy as np
import sys
sys.path.append('../general')
from metric import metric
from constants import POSITION_VARIANTS


def make_features(main_features, positions, one_hot=False):
    if not one_hot:
        return np.concatenate((main_features, np.reshape(positions, (-1, 1))), axis=1)
    else:
        repeated_pos_variants = np.repeat([POSITION_VARIANTS], len(main_features), axis=0)
        repeated_pos = np.repeat(np.reshape(positions, (-1, 1)), len(POSITION_VARIANTS), axis=1)
        return np.concatenate((main_features, (repeated_pos_variants == repeated_pos).astype(int)), axis=1)


def get_positions(features_to_answer, model, one_hot=False):
    positions = []
    for features in features_to_answer:
        features = np.repeat(np.reshape(features, (1, -1)), len(POSITION_VARIANTS), axis=0)
        predicted_scores = model.predict(make_features(features, POSITION_VARIANTS, one_hot))
        positions.append(np.argmax(predicted_scores))

        # mask = np.any(predicted_scores > 0, axis=1)
        # show_positions = np.argmax(predicted_scores, axis=1)
        # positions = show_positions * mask + NONE_POSITION * (1 - mask)
    return np.array(positions)


def get_metric(pool, model, one_hot=False):
    train_pool, test_pool = pool.train_test_split()
    model.fit(make_features(train_pool.features, train_pool.positions, one_hot), train_pool.targets)
    positions = get_positions(test_pool.features, model, one_hot)

    return metric(positions, test_pool.positions, test_pool.targets, test_pool.probas)


def test_regression(models, pool, one_hot=False):
    scores = [
        get_metric(pool, model, one_hot)
        for model in models
    ]
    return np.array(scores)
