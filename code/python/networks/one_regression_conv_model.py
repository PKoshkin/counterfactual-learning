import tensorflow as tf
from keras.layers import Conv1D
from keras import backend as Keras
import numpy as np

class OneRegressionConvModel:
    def __init__(self, num_features):
        self.num_features = num_features
        self.POSITIONS = list(range(11))
        self.input_features = tf.placeholder('float32', shape=(None, num_features, 1))
        self.input_prediction = tf.placeholder('float32', shape=(None, 1))
        conv = Conv1D(
            filters=8,
            kernel_size=100,
            activation='relu'
        )
        self.output_prediction = conv(self.input_features)
        self.loss = (
            tf.reduce_mean((self.input_prediction - self.output_prediction) ** 2)
        )
        self.optimizer = tf.train.AdamOptimizer().minimize(
            self.loss, var_list=conv.weights
        )

        self.sess = tf.Session()
        Keras.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def teach(self, train_features, train_prediction,
              verbose=True, iterations=100):
        for i in range(iterations):
            self.sess.run(
                self.optimizer, {
                    self.input_features: self.reshape_features(train_features),
                    self.input_prediction: train_prediction
                }
            )
            if verbose:
                train_loss = self.get_regression_loss(train_features, train_prediction)[0]
                print(
                    "train loss: {}".format(
                        train_loss
                    )
                )

    def reshape_features(self, features):
        return np.reshape(features, (len(features), self.num_features, 1))

    def predict_score(self, features):
        return self.sess.run(self.output_prediction, {self.input_features: self.reshape_features(features)})[0]

    def get_regression_loss(self, features, prediction):
         return self.sess.run(
            self.loss, {
                self.input_features: self.reshape_features(features),
                self.input_prediction: prediction
            }
        )

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
                ])
                scores_log.append(new_score)
                if new_score > max_score:
                    max_score = new_score
                    best_position = position
            if verbose:
                if (np.argmax(scores_log) + 1 == test_pool.positions[i]) and (test_pool.targets[i] != 0):
                    print(np.round(scores_log, 2), np.round(np.max(scores_log), 2), np.round(test_pool.targets[i], 2))
            prediction.append(best_position)
        return prediction
