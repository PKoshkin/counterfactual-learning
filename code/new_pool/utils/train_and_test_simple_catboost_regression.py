from __future__ import print_function
import argparse
import os
from time import time
from catboost import CatBoostRegressor, Pool
import numpy as np


def calculate_metric(answers_positions, target_positions, target, target_probs):
    mask = target_probs != 0
    return np.mean(
        target[mask] / target_probs[mask] * (answers_positions[mask] == target_positions[mask])
    )


def get_from_catboost_file(filename, indices, types, splitter='\t'):
    result = tuple([] for index in indices)
    with open(filename) as handler:
        for line in handler:
            line = line.split(splitter)
            for type_constructor, index in zip(types, indices):
                result[index].append(type_constructor(line[index]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    args = parser.parse_args()

    days_number = 16
    positions_number = 10

    pools = [
        Pool(os.path.join(args.data_folder, "day_{}.txt".format(i)))
        for i in xrange(days_number)
    ]

    models = [CatBoostRegressor() for _ in xrange(days_number - 1)]
    trains, tests = [], []
    for i in xrange(1, days_number):
        trains.append(range(i))
        tests.append(i)

    with open(os.path.join(args.out_folder, "times.txt"), 'w') as times_handler,\
         open(os.path.join(args.out_folder, "metrics.txt"), 'w') as metrics_handler:
        for i, (model, train, test) in enumerate(zip(models, trains, tests)):
            start = time()
            for train_index in train:
                model.fit(pools[train_index], verbose=True)
            end = time()
            train_time = end - start

            start = time()
            predicted_positions = []
            for feature in pools[test].get_features():
                positions = np.reshape(np.arange(positions_number), [positions_number, 1])
                repeated_feature = np.repeat(np.array([feature[1:]]), positions_number, axis=0)
                features_to_predict = np.concatenate([positions, repeated_feature], axis=1)
                predictions = model.predict(features_to_predict)
                predicted_positions.append(np.argmax(predictions))

            target_positions, targets = get_from_catboost_file(
                os.path.join(args.data_folder, "day_{}.txt".format(i)),
                indices=[0, 1],
                types=[int, int]
            )
            probas = get_from_catboost_file(
                os.path.join(args.data_folder, "info_{}.txt".format(i)),
                indices=[0],
                types=[float]
            )[0]
            metric = calculate_metric(predicted_positions, target_positions, targets, probas)
            end = time()
            predict_time = end - start
            print(metric, file=metrics_handler)
            print(train_time, predict_time, file=times_handler)


if __name__ == "__main__":
    main()
