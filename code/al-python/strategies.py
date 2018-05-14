# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from math import ceil
import sys


class BaseStrategy(object):
    def __init__(self, params):
        raise NotImplementedError()

    def get_batch_indexes(self, probs, labeled_pool, unlabeled_pool, batch_size):
        raise NotImplementedError()

    @property
    def params(self):
        return self._params

    @classmethod
    def get_info(cls, params):
        raise NotImplementedError()


class PassiveLearningStrategy(BaseStrategy):
    name = 'random'

    def __init__(self, params):
        self._params = {}

    def get_batch_indexes(self, probs, labeled_pool, unlabeled_pool, batch_size):
        if len(unlabeled_pool) < batch_size:
            return np.arange(len(unlabeled_pool))
        else:
            return np.random.choice(len(unlabeled_pool), size=batch_size, replace=False)

    @classmethod
    def get_info(cls, params):
        return params


class BaseActiveLearningStrategy(BaseStrategy):
    def get_batch_indexes(self, probs, labeled_pool, unlabeled_pool, batch_size):
        if batch_size >= len(unlabeled_pool):
            return np.arange(len(unlabeled_pool))

        active_batch_size = int(round(batch_size * (1 - self._random_part)))
        scores = self._get_scores(probs, labeled_pool, unlabeled_pool)
        indexes_to_label = np.argpartition(scores, -active_batch_size)[-active_batch_size:]
        if self._random_part > 0.0:
            random_batch_size = int(round(batch_size * self._random_part))
            all_indexes = np.arange(len(unlabeled_pool))
            not_chosen = np.delete(all_indexes, indexes_to_label)
            random_indexes_to_label = np.random.choice(
                not_chosen,
                size=random_batch_size,
                replace=False
            )
            indexes_to_label = np.concatenate([indexes_to_label, random_indexes_to_label])

        self._update_params(probs, labeled_pool, unlabeled_pool, indexes_to_label)
        return indexes_to_label

    def _get_scores(self, probs, labeled_pool, unlabeled_pool):
        raise NotImplementedError()

    def _update_params(self, probs, labeled_pool, unlabeled_pool, indexes_to_label):
        raise NotImplementedError()

    @property
    def _random_part(self):
        random_part = self._params.get('random_part', 0.0)
        assert random_part >= 0.0 and random_part < 1.0, 'random_part should be in [0, 1)'

        return random_part


class UncertaintySamplingActiveLearningStrategy(BaseActiveLearningStrategy):
    name = 'US'

    def __init__(self, params):
        self._params = params

    def _get_scores(self, probs, labeled_pool, unlabeled_pool):
        if self._uncertainty_metric == 'max':
            return 1 - np.max(probs, axis=1)

        elif self._uncertainty_metric == 'gini':
            return 1 - np.sum(probs * probs, axis=1)

        else:
            raise ValueError('Unknown uncertainty type: {}'.format(self._uncertainty_metric))

    def _update_params(self, probs, labeled_pool, unlabeled_pool, indexes_to_label):
        pass

    @classmethod
    def get_info(cls, params):
        return params

    @property
    def _uncertainty_metric(self):
        return self._params.get('uncertainty_metric', 'max')


class BaseDensityBasedActiveLearningStrategy(BaseActiveLearningStrategy):
    def __init__(self, params):
        self._params = params
        self._batch_size = 64

    def _aggregation(self, closeness_2d, axis):
        raise NotImplementedError()

    def _init_closeness(self, labeled_pool, unlabeled_pool):
        raise NotImplementedError()

    def _compute_scores(self, probs, labeled_pool, unlabeled_pool):
        raise NotImplementedError()

    def _update_closeness(self, new_unlabeled_pool, new_labeled_pool_part):
        raise NotImplementedError()

    def _get_scores(self, probs, labeled_pool, unlabeled_pool):
        if self._closeness is None:
            sys.stderr.write(
                'closeness is None, make sure this is the first' +
                'Active Learning Iteration cube with density based strategy\n'
            )
            self._init_closeness(labeled_pool, unlabeled_pool)

        return self._compute_scores(probs, labeled_pool, unlabeled_pool)

    def _compute_closeness(self, first_pool, second_pool):
        closeness = np.zeros(len(first_pool))
        reduced_size = max(int(len(second_pool) * self._share), 1)
        second_pool_ind = np.random.choice(len(second_pool), size=reduced_size, replace=False)
        second_pool = second_pool[second_pool_ind]

        for b_start_ind in range(0, len(first_pool), self._batch_size):
            b_end_ind = b_start_ind + self._batch_size
            tmp_closeness = np.dot(
                first_pool[b_start_ind:b_end_ind],
                second_pool.T
            )
            norm = np.linalg.norm(first_pool[b_start_ind:b_end_ind], axis=1)
            tmp_closeness = (tmp_closeness.T / norm).T
            tmp_closeness /= np.linalg.norm(second_pool, axis=1)
            closeness[b_start_ind:b_end_ind] = self._aggregation(tmp_closeness, axis=1)
        return closeness

    def _update_params(self, probs, labeled_pool, unlabeled_pool, indexes_to_label):
        mask = np.ones(unlabeled_pool.shape[0], bool)
        mask[indexes_to_label] = False
        self._closeness = self._closeness[mask]

        new_labeled_pool_part = unlabeled_pool[indexes_to_label]
        new_unlabeled_pool = unlabeled_pool[mask]
        self._update_closeness(new_unlabeled_pool, new_labeled_pool_part)

    @property
    def _share(self):
        return self._params.get('share', 1.0)

    @_share.setter
    def _share(self, share):
        if share > 1.0 or share <= 0:
            raise ValueError('share should be between 0 and 1')
        self._params['share'] = share

    @property
    def _closeness(self):
        return self._params.get('closeness', None)

    @_closeness.setter
    def _closeness(self, value):
        self._params['closeness'] = value

    @classmethod
    def get_info(cls, params):
        info = {
            key: value
            for key, value in params.items()
            if key != 'closeness'
        }
        return info


class DiversityActiveLearningStrategy(BaseDensityBasedActiveLearningStrategy):
    name = 'diversity'

    def _aggregation(self, closeness_2d, axis):
        return np.max(closeness_2d, axis=axis)

    def _init_closeness(self, labeled_pool, unlabeled_pool):
        self._closeness = self._compute_closeness(unlabeled_pool, labeled_pool)

    def _compute_scores(self, probs, labeled_pool, unlabeled_pool):
        return -self._closeness

    def _update_closeness(self, new_unlabeled_pool, new_labeled_pool_part):
        closeness_update = self._compute_closeness(new_unlabeled_pool, new_labeled_pool_part)
        self._closeness = np.maximum(closeness_update, self._closeness)


class DensityActiveLearningStrategy(BaseDensityBasedActiveLearningStrategy):
    name = 'density'

    def _aggregation(self, closeness_2d, axis):
        return np.mean(closeness_2d, axis=axis)

    def _init_closeness(self, labeled_pool, unlabeled_pool):
        self._closeness = self._compute_closeness(unlabeled_pool, unlabeled_pool)

    def _compute_scores(self, probs, labeled_pool, unlabeled_pool):
        return self._closeness ** self._beta

    def _update_closeness(self, new_unlabeled_pool, new_labeled_pool_part):
        closeness_update = self._compute_closeness(new_unlabeled_pool, new_labeled_pool_part)
        old_len = len(new_unlabeled_pool) + len(new_labeled_pool_part)
        new_len = len(new_unlabeled_pool)
        part_len = len(new_labeled_pool_part)

        self._closeness = (self._closeness * old_len - closeness_update * part_len) / new_len

    @property
    def _beta(self):
        return self._params.get('beta', 1.0)


class MixActiveLearningStrategy(BaseActiveLearningStrategy):
    def __init__(self, params):
        self._params = params
        self._strategies = [
            STRATEGIES[strategy_name](strategy_params)
            for strategy_name, strategy_params in self._params.items()
            if strategy_name in STRATEGIES
        ]

    def _get_scores(self, probs, labeled_pool, unlabeled_pool):
        scores = np.ones(len(unlabeled_pool))
        for strategy in self._strategies:
            scores *= strategy._get_scores(probs, labeled_pool, unlabeled_pool)
        return scores

    def _update_params(self, probs, labeled_pool, unlabeled_pool, indexes_to_label):
        for strategy in self._strategies:
            strategy._update_params(probs, labeled_pool, unlabeled_pool, indexes_to_label)

    @classmethod
    def get_info(cls, params):
        info = {}
        for key in params.keys():
            info.update({
                key + '_' + param_key: param_value
                for param_key, param_value in STRATEGIES[key].get_info(params[key]).items()
            })

        return info


STRATEGY_CLASSES = [
    UncertaintySamplingActiveLearningStrategy,
    PassiveLearningStrategy,
    DiversityActiveLearningStrategy,
    DensityActiveLearningStrategy,
]
STRATEGIES = {
    cls.name: cls
    for cls in STRATEGY_CLASSES
}


def check_existance(strategy):
    strategy_parts = strategy.split('-')
    for strategy_part in strategy_parts:
        if strategy_part not in STRATEGIES.keys():
            return False

    return True


def _preproc(strategy, params):
    assert check_existance(strategy), "Unknown strategy: {}".format(strategy)
    if '-' in strategy:
        for name in strategy.split('-'):
            params.setdefault(name, {})
        return MixActiveLearningStrategy, params
    else:
        return STRATEGIES[strategy], params


def get_info(strategy, params):
    strategy_cls, params = _preproc(strategy, params)
    return strategy_cls.get_info(params)


def get_strategy(strategy, params):
    strategy_cls, params = _preproc(strategy, params)
    return strategy_cls(params)
