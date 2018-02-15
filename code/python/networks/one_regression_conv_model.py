import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Activation
from keras import backend as Keras
import numpy as np

class OneRegressionConvModel:
    def __init__(self, num_features):
        self.num_features = num_features
        self.NONE_POSITION = 11
        self.POSITIONS = list(range(1, self.NONE_POSITION))

        self.sequential_model = Sequential([
            Conv1D(
                8, 5, activation='relu',
                use_bias=True,
                input_shape=(self.num_features, 1)
            ),
            Conv1D(
                16, 5, activation='relu',
                use_bias=True
            ),
            Conv1D(
                32, 5, activation='relu',
                use_bias=True
            ),
            Flatten(),
            Dense(512),
            Dense(1)
        ])
        self.sequential_model.compile(optimizer='adam', loss='mse')

    def reshape_features(self, features):
        return np.reshape(features, (len(features), self.num_features, 1))

    def teach(self, train_features, train_prediction,
              verbose=True, epochs=1, batch_size=32):
        self.sequential_model.fit(
            self.reshape_features(train_features), train_prediction,
            epochs=epochs, batch_size=batch_size
        )

    def predict_score(self, features):
        return self.sequential_model.predict(np.reshape(features, (len(features), -1, 1)))

    def predict_positions(self, test_pool, verbose=True):
        prediction = []
        for i, features in enumerate(test_pool.features):
            max_score = -100
            best_position = -100
            scores_log = []
            for position in self.POSITIONS:
                new_score = self.predict_score([
                    list(features) + [
                        0 if possible_position != position else 1
                        for possible_position in self.POSITIONS
                    ]
                ])[0]
                scores_log.append(new_score)
                if new_score > max_score:
                    max_score = new_score
                    best_position = position
            scores_log = np.array(scores_log)
            if verbose:
                if (np.argmax(scores_log) + 1 == test_pool.positions[i]) and (test_pool.targets[i] != 0):
                    print(np.round(scores_log, 2), np.round(np.max(scores_log), 2), np.round(test_pool.targets[i], 2))
            prediction.append(best_position)
        return prediction
