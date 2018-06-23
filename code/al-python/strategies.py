# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from math import ceil
import sys

from utils import get_features, POSITIONS, get_positions
from classifier import train, predict_positions


def _get_indexes(scores, unlabeled_pool, size, split_by_positions):
    if split_by_positions:
        pos_batch_size = int(round(size / len(POSITIONS)))
        indexes_by_pos = []
        for pos in POSITIONS:
            scores_for_pos = scores[get_positions(unlabeled_pool) == pos]
            pos_indexes = np.argpartition(scores_for_pos, -pos_batch_size)[-pos_batch_size:]
            indexes_by_pos.append(pos_indexes)
        indexes_to_label = np.concatenate(indexes_by_pos)
    else:
        indexes_to_label = np.argpartition(scores, -size)[-size:]

    return indexes_to_label


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
        if batch_size < len(unlabeled_pool):
            return np.random.choice(len(unlabeled_pool), size=batch_size, replace=False)
        else:
            return np.arange(len(unlabeled_pool))

    @classmethod
    def get_info(cls, params):
        return params


class BaseActiveLearningStrategy(BaseStrategy):
    def get_batch_indexes(self, probs, labeled_pool, unlabeled_pool, batch_size):
        if batch_size >= len(unlabeled_pool):
            return np.arange(len(unlabeled_pool))

        active_batch_size = int(round(batch_size * (1 - self._random_part)))
        scores = self._get_scores(probs, labeled_pool, unlabeled_pool, batch_size)

        if getattr(self, 'return_indexes', False):  # not the greatest idea of all times :(
            indexes_to_label = scores
        else:
            indexes_to_label = _get_indexes(
                scores, unlabeled_pool, active_batch_size,
                self._split_by_positions
            )

        if self._random_part > 0.0:
            random_batch_size = batch_size - len(indexes_to_label)
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

    def _get_scores(
            self, probs, labeled_pool, unlabeled_pool, batch_size,
            indexes_to_choose_from=None
            ):
        raise NotImplementedError()

    def _update_params(self, probs, labeled_pool, unlabeled_pool, indexes_to_label):
        raise NotImplementedError()

    @property
    def _random_part(self):
        random_part = self._params.get('random_part', 0.0)
        assert random_part >= 0.0 and random_part < 1.0, 'random_part should be in [0, 1)'

        return random_part

    @property
    def _split_by_positions(self):
        split_by_positions = self._params.get('split_by_positions', False)
        assert isinstance(split_by_positions, bool), 'split_by_positions should be bool'

        return split_by_positions


class ProbaBasedActiveLearningStrategy(BaseActiveLearningStrategy):

    def __init__(self, params):
        self._params = params

    def _update_params(self, probs, labeled_pool, unlabeled_pool, indexes_to_label):
        pass

    def _preprocess_pools(self, labeled_pool, unlabeled_pool):
        return labeled_pool, unlabeled_pool

    @classmethod
    def get_info(cls, params):
        info = {
            key: value
            for key, value in params.items()
            if key not in ['classes_num']
        }
        return info

    def _normalize_proba(self, proba):
        if self._proba_normalization == 'sum':
            return (proba.T / np.sum(proba, axis=1)).T

        if self._proba_normalization == 'softmax':
            exp = np.exp((proba.T - np.max(proba, axis=1)) / self._temperature).T
            return (exp.T / np.sum(exp, axis=1)).T

    @property
    def _classes_num(self):
        return self._params.get('classes_num', 11)

    @property
    def _proba_normalization(self):
        return self._params.get('proba_normalization', 'sum')

    @property
    def _temperature(self):
        return self._params.get('T', 0.1)


class PositionRelevanceActiveLearningStragety(ProbaBasedActiveLearningStrategy):
    name = 'PR'

    def _get_scores(
            self, probs, labeled_pool, unlabeled_pool, batch_size,
            indexes_to_choose_from=None
            ):
        if indexes_to_choose_from is not None:
            probs = probs[indexes_to_choose_from]
            unlabeled_pool = unlabeled_pool[indexes_to_choose_from]
        if self._relevance_metric == 'delta_max':
            max_values = np.max(probs, axis=1)
            rnd_values = probs[
                np.arange(len(probs)), unlabeled_pool[:, 1].astype(np.int) % self._classes_num
            ]
            delta = max_values - rnd_values
            return 1.0 / (self._delta_addition + delta)

    @property
    def _relevance_metric(self):
        return self._params.get('relevance_metric', 'delta_max')

    @property
    def _delta_addition(self):
        return self._params.get('delta_addition', 0.1)


class UncertaintySamplingActiveLearningStrategy(ProbaBasedActiveLearningStrategy):
    name = 'US'

    def _get_scores(
            self, probs, labeled_pool, unlabeled_pool, batch_size,
            indexes_to_choose_from=None
            ):
        if indexes_to_choose_from is not None:
            unlabeled_pool = unlabeled_pool[indexes_to_choose_from]
            probs = probs[indexes_to_choose_from]

        if self._uncertainty_metric == 'max':
            return 1 - np.max(probs, axis=1)

        elif self._uncertainty_metric == 'gini':
            probs = self._normalize_proba(probs)
            return 1 - np.sum(probs * probs, axis=1)

        elif self._uncertainty_metric == 'entropy':
            probs = self._normalize_proba(probs)
            return -np.sum(probs * np.log(probs + 1e-10), axis=1)

        elif self._uncertainty_metric == 'delta':
            partitioned_array = np.partition(probs, self._classes_num - 2, axis=1)
            max_values = partitioned_array[:, -1]
            pre_max_values = partitioned_array[:, -2]
            delta = max_values - pre_max_values
            return 1.0 / (self._delta_addition + delta)

        else:
            raise ValueError('Unknown uncertainty type: {}'.format(self._uncertainty_metric))

    def _update_params(self, probs, labeled_pool, unlabeled_pool, indexes_to_label):
        pass

    @property
    def _delta_addition(self):
        return self._params.get('delta_addition', 0.1)

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

    def _compute_scores(self, probs, labeled_pool, unlabeled_pool, indexes_to_choose_from):
        raise NotImplementedError()

    def _update_closeness(self, new_unlabeled_pool, new_labeled_pool_part, indexes_to_label):
        raise NotImplementedError()

    def _get_scores(
            self, probs, labeled_pool, unlabeled_pool, batch_size,
            indexes_to_choose_from=None
            ):
        if self._closeness is None:
            sys.stderr.write(
                'closeness is None, make sure this is the first ' +
                'active learning iteration with density based strategy\n'
            )
            self._init_closeness(labeled_pool, unlabeled_pool)

        return self._compute_scores(probs, labeled_pool, unlabeled_pool, indexes_to_choose_from)

    def _compute_closeness(self, first_pool, second_pool, first_pool_start=0):
        closeness = np.zeros(len(first_pool) - first_pool_start)
        reduced_size = max(int(len(second_pool) * self._share), 1)
        second_pool_ind = np.random.choice(len(second_pool), size=reduced_size, replace=False)
        second_pool = get_features(second_pool[second_pool_ind], add_position=False)
        first_pool = get_features(first_pool, add_position=False)

        for b_start_ind in range(first_pool_start, len(first_pool), self._batch_size):
            b_end_ind = b_start_ind + self._batch_size
            tmp_closeness = np.dot(
                first_pool[b_start_ind:b_end_ind],
                second_pool.T
            )
            norm = np.linalg.norm(first_pool[b_start_ind:b_end_ind], axis=1)
            tmp_closeness = (tmp_closeness.T / norm).T
            tmp_closeness /= np.linalg.norm(second_pool, axis=1)

            closeness_start = b_start_ind - first_pool_start
            closeness_end = b_end_ind - first_pool_start
            closeness[closeness_start:closeness_end] = self._aggregation(tmp_closeness, axis=1)
        return closeness

    def _update_params(self, probs, labeled_pool, unlabeled_pool, indexes_to_label):
        mask = np.ones(unlabeled_pool.shape[0], bool)
        mask[indexes_to_label] = False
        self._closeness = self._closeness[mask]

        new_labeled_pool_part = unlabeled_pool[indexes_to_label]
        new_unlabeled_pool = unlabeled_pool[mask]
        self._update_closeness(new_unlabeled_pool, new_labeled_pool_part, indexes_to_label)

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

    def _compute_scores(self, probs, labeled_pool, unlabeled_pool, indexes_to_choose_from):
        if indexes_to_choose_from is None:
            closeness = self._closeness
        else:
            closeness = self._closeness[indexes_to_choose_from]
        return 1 - closeness

    def _update_closeness(self, new_unlabeled_pool, new_labeled_pool_part, indexes_to_label):
        closeness_update = self._compute_closeness(new_unlabeled_pool, new_labeled_pool_part)
        self._closeness = np.maximum(closeness_update, self._closeness)


class DensityActiveLearningStrategy(BaseDensityBasedActiveLearningStrategy):
    name = 'density'

    def _aggregation(self, closeness_2d, axis):
        return np.mean(closeness_2d, axis=axis)

    def _init_closeness(self, labeled_pool, unlabeled_pool):
        if self._all_data:
            all_data = np.concatenate([labeled_pool, unlabeled_pool], axis=0)
            self._closeness = self._compute_closeness(all_data, all_data, len(labeled_pool))
        else:
            self._closeness = self._compute_closeness(unlabeled_pool, unlabeled_pool)

    def _compute_scores(self, probs, labeled_pool, unlabeled_pool, indexes_to_choose_from):
        if indexes_to_choose_from is None:
            closeness = self._closeness
        else:
            closeness = self._closeness[indexes_to_choose_from]
        return 1 + closeness ** self._beta

    def _update_closeness(self, new_unlabeled_pool, new_labeled_pool_part, indexes_to_label):
        if not self._all_data:
            closeness_update = self._compute_closeness(new_unlabeled_pool, new_labeled_pool_part)
            old_len = len(new_unlabeled_pool) + len(new_labeled_pool_part)
            new_len = len(new_unlabeled_pool)
            part_len = len(new_labeled_pool_part)

            self._closeness = (self._closeness * old_len - closeness_update * part_len) / new_len

    @property
    def _beta(self):
        return self._params.get('beta', 1.0)

    @property
    def _all_data(self):
        return self._params.get('all_data', False)


class MixTypes(object):
    MULTIPLY = 'multiply'
    EXPLORATION_GUIDED = 'EG'

    @classmethod
    def get_all(cls):
        return [
            getattr(cls, var)
            for var in vars(cls)
            if not var.startswith('__') and var != 'get_all'
        ]


class MixActiveLearningStrategy(BaseActiveLearningStrategy):
    def __init__(self, params):
        self._params = params
        assert self._strategies is not None, 'params should contain strategies'
        self._strategies = [
            STRATEGIES[strategy](self._params[strategy])
            for strategy in self._strategies
        ]
        if self._mix_type == MixTypes.EXPLORATION_GUIDED:
            self.return_indexes = True

    def _get_scores(
            self, probs, labeled_pool, unlabeled_pool, batch_size,
            indexes_to_choose_from=None
            ):
        if self._mix_type == MixTypes.MULTIPLY:
            scores = np.ones(len(unlabeled_pool))
            for strategy in self._strategies:
                scores *= strategy._get_scores(
                    probs, labeled_pool, unlabeled_pool, batch_size,
                    indexes_to_choose_from
                )
            return scores
        elif self._mix_type == MixTypes.EXPLORATION_GUIDED:
            batches = [
                int(batch_size * self._reserve_size ** i)
                for i in xrange(len(self._strategies) - 1, -1, -1)
            ]
            chosed_indexes = np.arange(len(unlabeled_pool))
            for batch, strategy in zip(batches, self._strategies):
                if batch >= len(unlabeled_pool):
                    sys.stderr.wirte(
                        'Batch size with reserve is bigger than unlabeled_pool size. ' +
                        'Probably, reserve_size is too big, consider reducing it. ' +
                        'Skipping {} strategy.\n'.format(strategy.name)
                    )
                    continue
                scores = strategy._get_scores(
                    probs, labeled_pool, unlabeled_pool, batch,
                    chosed_indexes
                )
                new_indexes = _get_indexes(
                    scores, unlabeled_pool[chosed_indexes], batch,
                    self._split_by_positions
                )
                chosed_indexes = chosed_indexes[new_indexes]

            return chosed_indexes

    @property
    def _strategies(self):
        return self._params.get('strategies')

    @_strategies.setter
    def _strategies(self, value):
        self._params['strategies'] = value

    def _update_params(self, probs, labeled_pool, unlabeled_pool, indexes_to_label):
        for strategy in self._strategies:
            strategy._update_params(probs, labeled_pool, unlabeled_pool, indexes_to_label)

    @classmethod
    def get_info(cls, params):
        info = {}
        for key, value in params.items():
            if key in STRATEGIES:
                info[key] = STRATEGIES[key].get_info(value)
            elif key != 'strategies':
                info[key] = value

        return info

    @property
    def _mix_type(self):
        mix_type = self._params.get('mix_type', MixTypes.MULTIPLY)
        assert mix_type in MixTypes.get_all(), (
            'Unknown type: {}, allowed types: {}'.format(mix_type, MixTypes.get_all())
        )
        return mix_type

    @property
    def _reserve_size(self):
        return self._params.get('reserve_size', 1.1)


class QBCMetrics(object):
    VE = 'VE'
    KL = 'KL'

    @classmethod
    def get_all(cls):
        return [
            getattr(cls, var)
            for var in vars(cls)
            if not var.startswith('__') and var != 'get_all'
        ]


class QBCActiveLearningStrategy(BaseActiveLearningStrategy):
    name = 'QBC'

    def __init__(self, params):
        self._params = params
        self._iteration = 0

    def _compute_metric(self, auxilary_predicted_probs, main_predicted_probs):
        all_probs = np.stack(auxilary_predicted_probs + [main_predicted_probs])
        num_models, num_objects, num_classes = all_probs.shape
        total_voutes_number = all_probs.shape[0] * self._voute_number

        if self._metric == QBCMetrics.VE:
            model_voute_indexes = np.argpartition(
                all_probs, -self._voute_number, axis=2
            )[:, :, -self._voute_number:]
            aggregated_voutes = np.zeros((num_objects, num_classes))
            indexes = np.repeat(np.arange(num_objects), self._voute_number)
            for model_ind in xrange(num_models):
                aggregated_voutes[indexes, model_voute_indexes[model_ind].reshape(-1)] += 1
            normed_voutes = aggregated_voutes / total_voutes_number
            return np.sum(-normed_voutes * np.log(normed_voutes + 1e-5), axis=1)

        if self._metric == QBCMetrics.KL:
            normed_probs /= np.sum(all_probs, axis=2)
            mean_probs = np.mean(normed_probs, axis=0)
            KL_distances = np.sum(normed_probs * np.log(normed_probs / mean_probs + 1e-5), axis=-1)
            return np.mean(KL_distances, axis=0)

    def _get_scores(
            self, probs, labeled_pool, unlabeled_pool, batch_size,
            indexes_to_choose_from=None
            ):
        if indexes_to_choose_from is not None:
            probs = probs[indexes_to_choose_from]
            unlabeled_pool = unlabeled_pool[indexes_to_choose_from]

        if self._iteration == 0:
            self._pool_indexes = [
                np.random.choice(len(labeled_pool), len(labeled_pool))
                for _ in xrange(self._models_num)
            ]
            self._models = [
                train(labeled_pool[indexes], verbose=False, **self._learning_params)
                for indexes in self._pool_indexes
            ]

        auxilary_predicted_probs = [
            predict_positions(unlabeled_pool, model, return_probs=True)
            for model in self._models
        ]
        return self._compute_metric(auxilary_predicted_probs, probs)

    def _update_params(self, probs, labeled_pool, unlabeled_pool, indexes_to_label):
        for pool_ind in xrange(self._models_num):
            new_indexes = len(labeled_pool) + np.random.choice(
                len(indexes_to_label), len(indexes_to_label)
            )
            self._pool_indexes[pool_ind] = np.concatenate((
                self._pool_indexes[pool_ind], new_indexes
            ))
        model_ind = self._iteration % self._models_num
        train_pool = np.concatenate(
            (
                labeled_pool[self._pool_indexes[model_ind][:len(labeled_pool)]],
                unlabeled_pool[indexes_to_label][new_indexes - len(labeled_pool)]
            ),
            axis=0
        )
        self._models[model_ind] = train(train_pool, verbose=False, **self._learning_params)

    @classmethod
    def get_info(cls, params):
        return params

    @property
    def _models_num(self):
        models_num = self._params.get('models_num', 3)

        assert isinstance(models_num, int) and models_num > 0, (
            'models_num should be int and greater than one. Got: ' + str(models_num)
        )
        return models_num

    @property
    def _learning_params(self):
        return self._params.get('learning_params', {})

    @property
    def _metric(self):
        metric = self._params.get('metric', QBCMetrics.VE)
        assert metric in QBCMetrics.get_all(), (
            'Unknown metric: {}, allowed metrics: {}'.format(metric, QBCMetrics.get_all())
        )
        return metric

    @property
    def _voute_number(self):
        voute_number = self._params.get('voute_number', 3)
        assert isinstance(voute_number, int) and voute_number > 0, (
            'voute_number should be int and greater than 0. Got: ' + str(voute_number)
        )
        return voute_number


STRATEGY_CLASSES = [
    UncertaintySamplingActiveLearningStrategy,
    PassiveLearningStrategy,
    DiversityActiveLearningStrategy,
    DensityActiveLearningStrategy,
    PositionRelevanceActiveLearningStragety,
    QBCActiveLearningStrategy,
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
        strategies = strategy.split('-')
        params['strategies'] = strategies
        for name in strategies:
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
