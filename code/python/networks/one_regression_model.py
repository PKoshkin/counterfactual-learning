import tensorflow as tf
import numpy as np

class OneRegressionModel:
    def __init__(self, num_features):
        self.POSITIONS = list(range(11))
        self.matrix = tf.get_variable(
            "matrix", shape=(num_features, 1),
            initializer=tf.glorot_uniform_initializer()
        )
        self.bias = tf.get_variable(
            "bias", shape=(1,),
            initializer=tf.glorot_uniform_initializer()
        )
        self.input_features = tf.placeholder('float32', shape=(None, num_features))
        self.input_prediction = tf.placeholder('float32', shape=(None, 1))
        self.output_prediction = tf.matmul(self.input_features, self.matrix) + self.bias
        self.loss = (
            tf.reduce_mean((self.input_prediction - self.output_prediction) ** 2) +
            tf.reduce_mean(self.matrix ** 2) + tf.reduce_mean(self.bias ** 2)
        )
        self.optimizer = tf.train.AdamOptimizer().minimize(
            self.loss, var_list=[self.matrix, self.bias]
        )

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.best_marix = self.sess.run(self.matrix)
        self.best_bias = self.sess.run(self.bias)
        
    def teach(self, train_features, train_prediction,
              test_features, test_prediction, verbose=True, iterations=100):
        best_test_loss = 1e111
        for i in range(iterations):
            self.sess.run(
                self.optimizer, {
                    self.input_features: train_features,
                    self.input_prediction: train_prediction
                }
            )
            test_loss = self.get_regression_loss(test_features, test_prediction)
            train_loss = self.get_regression_loss(train_features, train_prediction)
            test_loss = test_loss[0]
            train_loss = train_loss[0]
            if test_loss < best_test_loss:
                self.best_marix = self.sess.run(self.matrix)
                self.best_bias = self.sess.run(self.bias)
                best_test_loss = test_loss
            if verbose:
                print(
                    "train loss: {}, test loss: {}".format(
                        train_loss, test_loss
                    )
                )

    def predict_score(self, features):
        return self.sess.run(self.output_prediction, {self.input_features: features})

    def get_regression_loss(self, features, prediction):
         return self.sess.run(
            self.loss, {
                self.input_features: features,
                self.input_prediction: prediction
            }
        ),

    def predict_positions(self, test_features):
        prediction = []
        for features in test_features:
            max_score = -100
            best_position = -100
            for position in self.POSITIONS:
                new_score = self.predict_score([
                    list(features) + [
                        0 if possible_position != position else 1
                        for possible_position in self.POSITIONS
                    ]
                ])
                new_score = new_score[0]
                print(new_score)
                if new_score > max_score:
                    max_score = new_score
                    best_position = position
            prediction.append(best_position)
        return prediction
