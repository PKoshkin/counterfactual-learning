#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import numpy as np
from math import ceil

from classifier import train, predict_positions
from utils import write_to_csv, read_csv, calculate_metric, log
from strategies import get_strategy, STRATEGIES


def _run_al_experiment(
    labeled_pool,
    unlabeled_pool,
    val_pool,
    test_pool,
    algo,
    strategy_name,
    strategy_params,
    strategy,
    batch_size,
    end_size,
    **classifier_params
):
    train_len = len(unlabeled_pool) + len(labeled_pool)
    batch_size = int(ceil(batch_size * train_len))
    if end_size == -1:
        end_size = train_len
    else:
        end_size = int(ceil(end_size * train_len))
    iters_num = int(ceil((float(end_size - len(labeled_pool))) / batch_size))
    params = {
        "algo": algo,
        "strategy": strategy_name,
        "strategy_params": json.dumps(strategy_params),
        "batch size": batch_size,
        "initial size": len(labeled_pool),
        "max queries": end_size,
        "train pool size": len(labeled_pool) + len(unlabeled_pool),
        "val pool size": len(val_pool),
        "test pool size": len(test_pool),
    }
    result = "Algorithm with following features was applied:\n"
    result += '\n'.join(["{}: {}".format(key, value) for key, value in params.items()])

    result += '\n\n'

    classifiers = []
    metrics = []

    for iteration_ind in xrange(iters_num + 1):
        log('begin {} iteration\n'.format(iteration_ind))
        classifier = train(labeled_pool, **classifier_params)

        if iteration_ind < iters_num:
            probs = predict_positions(unlabeled_pool, classifier, return_probs=True)
            indexes = strategy.get_batch_indexes(probs, labeled_pool, unlabeled_pool, batch_size)
            labeled_pool = np.concatenate([labeled_pool, unlabeled_pool[indexes]], axis=0)
            unlabeled_pool = np.delete(unlabeled_pool, indexes, axis=0)

        predicted_positions = predict_positions(val_pool, classifier)
        metric = calculate_metric(predicted_positions, val_pool)

        result += str(metric) + '\n'
        metrics.append(metric)
        classifiers.append(classifier)

    best_iteration = np.argmax(metrics)
    best_classifier = classifiers[best_iteration]
    predicted_positions = predict_positions(test_pool, best_classifier)
    final_metric = calculate_metric(predicted_positions, test_pool)
    result += 'best iteration: ' + str(best_iteration) + '\n'
    result += 'metric on test using classifier from best iteration: ' + str(final_metric) + '\n'
    result += '\n'
    print(result)
    return result


def _get_initial_pools(train_pool, random_seed, initial_size):
    """Return [labeled_pool, unlabeled_pool]"""
    np.random.seed(random_seed)
    np.random.shuffle(train_pool)
    return np.split(train_pool, [initial_size])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pool', required=True)
    parser.add_argument('--val_pool', required=True)
    parser.add_argument('--test_pool', required=True)
    parser.add_argument('--random_seed', required=True, type=int)
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--initial_size', required=True, type=float)
    parser.add_argument('--batch_size', required=True, type=float)
    parser.add_argument('--end_size', default=-1, type=float)
    parser.add_argument('--strategy_params', type=json.loads, default={})
    parser.add_argument('--results', required=True)
    parser.add_argument('--train_steps', type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    train_pool = read_csv(args.train_pool)
    val_pool = read_csv(args.val_pool)
    test_pool = read_csv(args.test_pool)
    initial_size = int(ceil(args.initial_size * len(train_pool)))
    labeled_pool, unlabeled_pool = _get_initial_pools(
        train_pool,
        args.random_seed,
        initial_size
    )

    strategy = get_strategy(args.strategy, args.strategy_params)

    result = _run_al_experiment(
        labeled_pool,
        unlabeled_pool,
        val_pool,
        test_pool,
        'pool-based',
        args.strategy,
        args.strategy_params,
        strategy,
        args.batch_size,
        args.end_size,
        verbose=False,
        iterations=args.train_steps,
    )
    with open(args.results, 'w') as handler:
        handler.write(result)


if __name__ == '__main__':
    main()
