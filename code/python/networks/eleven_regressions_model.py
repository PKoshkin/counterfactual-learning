import tensorflow as tf
import numpy as np

from one_position_model import OnePositionModel

class ElevenRegressionsModel:
    def __init__(self, num_features):
        self.NONE_POSITION = 11
        self.POSITIONS = list(range(1, self.NONE_POSITION))
        self.models = [
            OnePositionModel(num_features, position)
            for position in self.POSITIONS
        ]

    def teach(self, train_pool, test_pool, verbose=True, iterations=1000):
        train_pools = train_pool.split_by_position()
        test_pools = test_pool.split_by_position()
        for i, model in enumerate(self.models):
            train_log, test_log = model.teach(
                train_pools[i].features, train_pools[i].regression_prediction,
                test_pools[i].features, test_pools[i].regression_prediction,
            )
            counter = 0
            best_test_loss = np.min(test_log)
            while best_test_loss > 1:
                train_log, test_log = model.teach(
                    train_pools[i].features, train_pools[i].regression_prediction,
                    test_pools[i].features, test_pools[i].regression_prediction,
                )
                best_test_loss = np.min(test_log)
                counter += 1
                if counter > 20:
                    break
            if verbose:
                print('position {}: {} ({} try)'.format(i + 1, np.min(test_log), counter))

    def predict_positions(self, test_features, verbose=True):
        prediction = []
        for features in test_features:
            max_score = -100
            best_position = -100
            scores_log = []
            for model, position in zip(self.models, self.POSITIONS):
                new_score = model.predict_score([features])[0]
                scores_log.append(new_score)
                if new_score > max_score:
                    max_score = new_score
                    best_position = position
            if verbose:
                print(scores_log, np.max(scores_log))
            if max_score > 0:
                prediction.append(best_position)
            else:
                prediction.append(self.NONE_POSITION)
        return prediction
