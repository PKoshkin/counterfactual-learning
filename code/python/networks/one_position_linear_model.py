import tensorflow as tf
import numpy as np

class OnePositionLinearModel:
    def __init__(self, num_features, position):
        self.matrix = tf.get_variable(
            name='matrix_' + str(position), shape=(num_features, 1),
            initializer=tf.glorot_uniform_initializer()
        )
        self.bias = tf.get_variable(
            name='bias_' + str(position), shape=(1,),
            initializer=tf.glorot_uniform_initializer()
        )
        self.input_features = tf.placeholder('float32', shape=(None, num_features))
        self.input_prediction = tf.placeholder('float32', shape=(None, 1))
        self.output_prediction = tf.matmul(self.input_features, self.matrix) + self.bias
        self.loss = (
            tf.reduce_mean((self.input_prediction - self.output_prediction) ** 2)# +
#            tf.reduce_mean(self.matrix ** 2) + tf.reduce_mean(self.bias ** 2)
        )
        self.optimizer = tf.train.AdamOptimizer().minimize(
            self.loss, var_list=[self.matrix, self.bias]
        )
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def teach(self, train_features, train_prediction, iterations=100):
        best_test_loss = 1e111
        train_losses_log = []
        for i in range(iterations):
            self.sess.run(
                self.optimizer, {
                    self.input_features: train_features,
                    self.input_prediction: train_prediction
                }
            )
            train_loss = self.get_regression_loss(train_features, train_prediction)
            train_losses_log.append(train_loss)
        return np.array(train_losses_log)

    def predict_score(self, features):
        return self.sess.run(self.output_prediction, {self.input_features: features})[0]

    def get_regression_loss(self, features, prediction):
         return self.sess.run(
            self.loss, {
                self.input_features: features,
                self.input_prediction: prediction
            }
        )
