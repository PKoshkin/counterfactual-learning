import tensorflow as tf
import numpy as np

class ElevenRegressionsModel:
    def __init__(self, num_features, OnePositionModel):
        self.NONE_POSITION = 11
        self.POSITIONS = list(range(1, self.NONE_POSITION))
        self.models = [
            OnePositionModel(num_features, position)
            for position in self.POSITIONS
        ]

    def teach(self, train_pool, verbose=True, iterations=1000):
        train_pools = train_pool.split_by_position()
        for i, model in enumerate(self.models):
            train_log = model.teach(
                train_pools[i].features, train_pools[i].regression_prediction, iterations
            )
            if verbose:
                print(' position loss {}: {}'.format(i + 1, train_log[-1]))

    def predict_positions(self, test_pool, verbose=True):
        prediction = []
        for i, features in enumerate(test_pool.features):
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
                if (np.argmax(scores_log) + 1 == test_pool.positions[i]) and (test_pool.targets[i] != 0):
                    print(np.round(scores_log, 2), np.round(np.max(scores_log), 2), np.round(test_pool.targets[i], 2))
            if max_score > 0:
                prediction.append(best_position)
            else:
                prediction.append(self.NONE_POSITION)

        return prediction
