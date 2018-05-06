# -*- coding: utf-8 -*-

from catboost import CatBoostClassifier
from utils import get_features, get_targets, get_weights
from utils import POSITIONS
from catboost import Pool
import numpy as np


def _raw_pool_to_catboost_train(pool):
    features = get_features(pool, add_position=True)
    return Pool(features, label=get_targets(pool) > 0, weight=get_weights(pool))


def train(train_pool, **classifier_params):
    classifier = CatBoostClassifier(**classifier_params)
    train_pool = _raw_pool_to_catboost_train(train_pool)
    classifier.fit(train_pool)
    return classifier


def predict_positions(test_pool, classifier, return_probs=False):
    probs = np.zeros((len(test_pool), len(POSITIONS)))

    for pos_ind, pos in enumerate(POSITIONS):
        test_features = get_features(test_pool, add_position=False)
        position_repeat = np.repeat(pos, len(test_pool)).reshape(-1, 1)
        test_features = np.concatenate([position_repeat, test_features], axis=1)
        probs[:, pos_ind] = classifier.predict_proba(test_features)[:, 1]

    if return_probs:
        return probs
    else:
        return np.array([POSITIONS[index] for index in np.argmax(probs, axis=1)])
