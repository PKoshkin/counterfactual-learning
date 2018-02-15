from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense
import numpy as np


class OnePositionConvModel:
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
            Dense(256),
            Dense(1)
        ])
        self.sequential_model.compile(optimizer='adam', loss='mse')

    def teach(self, train_features, train_prediction,
              verbose, epochs, batch_size):
        self.sequential_model.fit(
            self.reshape_features(train_features), train_prediction,
            epochs=epochs, batch_size=batch_size, verbose=verbose
        )

    def predict_score(self, features):
        return self.sequential_model.predict(np.reshape(features, (len(features), -1, 1)))

    def reshape_features(self, features):
        return np.reshape(features, (len(features), self.num_features, 1))
