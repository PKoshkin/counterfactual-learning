import numpy as np
import sys
sys.path.append('../general')
from metric import metric
from constants import POSITION_VARIANTS, NONE_POSITION


def test_classification(model, pool, runs=100):
    scores = []
    for i in range(runs):
        train_pool, test_pool = pool.train_test_split()
        model.fit(train_pool.features_with_positions, train_pool.classification_labels)

        repeated_features = np.repeat(test_pool.features, len(POSITION_VARIANTS), axis=0)
        repeated_positions = np.repeat([POSITION_VARIANTS], len(test_pool.features), axis=0).flatten()
        features_with_positions = np.concatenate((repeated_features, np.reshape(repeated_positions, (-1, 1))), axis=1)

        predicted_scores = model.predict_proba(features_with_positions)[:,2]
        predicted_scores = np.reshape(predicted_scores, (len(test_pool.features), len(POSITION_VARIANTS)))

        mask = np.any(predicted_scores > 0, axis=1)
        show_position = np.argmax(predicted_scores, axis=1)
        positions = show_position * mask + NONE_POSITION * (1 - mask)
        scores.append(metric(positions, test_pool.positions, test_pool.targets, test_pool.probas))
    return scores
