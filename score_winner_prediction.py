import tensorflow as tf
import numpy as np
import pandas as pd


class Model:
    def __init__(self, name):
        tf.set_random_seed(777)
        self.name = name
        self.xavier = tf.contrib.layers.xavier_initializer()
        self._learning_rate = 0.01
        self._training_epoch = 1000
        self._sess = None
        self.hypothesis = None
        self.X_home = None
        self.X_away = None
        self.Y = None
        self.keep_prob = None
        self.cost = None
        self.optimizer = None

    # region [Functions]
    def get_weight(self, name, w_size):
        return tf.get_variable(name, w_size, initializer=self.xavier)

    @staticmethod
    def get_bias(name, b_size):
        return tf.Variable(tf.random_normal(b_size), name=name)

    @staticmethod
    def get_layer(act, w_in, w_out, b, drop=None):
        layer = act(tf.add(tf.matmul(w_in, w_out), b))
        if drop:
            layer = tf.nn.dropout(layer, keep_prob=drop)
        return layer
    # endregion [Functions]

    # region [Hyper Params]
    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, rate):
        self._learning_rate = rate

    @property
    def training_epoch(self):
        return self._training_epoch

    @training_epoch.setter
    def training_epoch(self, num):
        self._training_epoch = num

    @property
    def sess(self):
        return self._sess

    @sess.setter
    def sess(self, session):
        self._sess = session
    # endregion [Hyper Params]

    # region [TF Board]
    @staticmethod
    def variable_summaries(var: object):
        """텐서에 많은 양의 요약정보(summaries)를 붙인다. (텐서보드 시각화를 위해서)"""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(var, mean))))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    # endregion [TF Board]

    # region [NN]
    def builder(self, input_size, output_size, model_name, act=tf.nn.relu):
        with tf.name_scope('input'):
            self.X_home = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
            self.X_away = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
            self.Y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

        with tf.name_scope(model_name):
            with tf.name_scope('dropout'):
                self.keep_prob = tf.placeholder(dtype=tf.float32)
                tf.summary.scalar('dropout_keep_probability', self.keep_prob)

            n_merged = input_size * 2 * 2
            n_neurons_1 = input_size * 2 * 2 * 10

            with tf.name_scope('weights'):
                weights = {
                    'w1_home': self.get_weight('W1_home_hidden', [input_size, input_size * 2]),
                    'w2_home': self.get_weight('W2_home_hidden', [input_size * 2, input_size]),
                    'w3_home': self.get_weight('W3_home_hidden', [input_size, input_size * 2]),
                    'w1_away': self.get_weight('W1_away_hidden', [input_size, input_size * 2]),
                    'w2_away': self.get_weight('W2_away_hidden', [input_size * 2, input_size]),
                    'w3_away': self.get_weight('W3_away_hidden', [input_size, input_size * 2]),
                    'w1': self.get_weight('W1_hidden', [n_merged, n_neurons_1]),
                    'w2': self.get_weight('W2_hidden', [n_neurons_1, n_neurons_1]),
                    'w3': self.get_weight('W3_hidden', [n_neurons_1, n_merged]),
                    'w4': self.get_weight('W4_hidden', [n_merged, output_size]),
                }
                for i, w in weights.items():
                    self.variable_summaries(w)

            with tf.name_scope('biases'):
                biases = {
                    'b1_home': self.get_bias('b1_home', [input_size * 2]),
                    'b2_home': self.get_bias('b2_home', [input_size]),
                    'b3_home': self.get_bias('b3_home', [input_size * 2]),
                    'b1_away': self.get_bias('b1_away', [input_size * 2]),
                    'b2_away': self.get_bias('b2_away', [input_size]),
                    'b3_away': self.get_bias('b3_away', [input_size * 2]),
                    'b1': self.get_bias('b1', [n_neurons_1]),
                    'b2': self.get_bias('b2', [n_neurons_1]),
                    'b3': self.get_bias('b3', [n_merged]),
                    'b4': self.get_bias('b4', [output_size]),
                }
                for i, bias in biases.items():
                    self.variable_summaries(bias)

            with tf.name_scope('layers'):
                h_l_1 = self.get_layer(act, self.X_home, weights['w1_home'], biases['b1_home'], self.keep_prob)
                h_l_2 = self.get_layer(act, h_l_1, weights['w2_home'], biases['b2_home'], self.keep_prob)
                h_l_3 = self.get_layer(act, h_l_2, weights['w3_home'], biases['b3_home'], self.keep_prob)
                tf.summary.histogram('home_hidden_1_activation', h_l_1)
                tf.summary.histogram('home_hidden_2_activation', h_l_2)
                tf.summary.histogram('home_hidden_3_activation', h_l_3)

                a_l_1 = self.get_layer(act, self.X_away, weights['w1_away'], biases['b1_away'], self.keep_prob)
                a_l_2 = self.get_layer(act, a_l_1, weights['w2_away'], biases['b2_away'], self.keep_prob)
                a_l_3 = self.get_layer(act, a_l_2, weights['w3_away'], biases['b3_away'], self.keep_prob)
                tf.summary.histogram('away_hidden_1_activation', a_l_1)
                tf.summary.histogram('away_hidden_2_activation', a_l_2)
                tf.summary.histogram('away_hidden_3_activation', a_l_3)

                merge_layers = tf.concat([a_l_3, h_l_3], 1)

                l_1 = self.get_layer(act, merge_layers, weights['w1'], biases['b1'], self.keep_prob)
                l_2 = self.get_layer(act, l_1, weights['w2'], biases['b2'], self.keep_prob)
                l_3 = self.get_layer(act, l_2, weights['w3'], biases['b3'], self.keep_prob)
                tf.summary.histogram('layer1_activation', l_1)
                tf.summary.histogram('layer2_activation', l_2)
                tf.summary.histogram('layer3_activation', l_3)

            with tf.name_scope('hypothesis'):
                self.hypothesis = tf.add(tf.matmul(l_3, weights['w4']), biases['b4'])
                tf.summary.histogram('hypothesis', self.hypothesis)

        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(tf.square(self.hypothesis - self.Y))
            tf.summary.scalar('cost', self.cost)
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self.cost)

    def get_accuracy(self, home_x_data, away_x_data, y_data, keep_prob=1.0):
        data_feed = {self.X_home: home_x_data, self.X_away: away_x_data, self.Y: y_data, self.keep_prob: keep_prob}
        predict = self.sess.run(self.hypothesis, feed_dict=data_feed)

        prediction = tf.map_fn(
            lambda x: x[0] > x[1],
            predict,
            dtype=bool
        )

        answer = tf.map_fn(
            lambda x: x[0] > x[1],
            y_data,
            dtype=bool
        )

        accuracy = tf.divide(tf.reduce_sum(tf.cast(tf.equal(prediction, answer), dtype=tf.int32)), len(y_data))
        return self.sess.run(accuracy)

    def train(self, home_x_train, away_x_train, y_train, keep_prob):
        train_feed = {self.X_home: home_x_train, self.X_away: away_x_train, self.Y: y_train, self.keep_prob: keep_prob}
        _cost, _opt = self.sess.run([self.cost, self.optimizer], feed_dict=train_feed)
        return _cost, _opt

    def predict(self, home_x_data, away_x_data, keep_prob=1.0):
        feed = {self.X_home: home_x_data, self.X_away: away_x_data, self.keep_prob: keep_prob}
        prediction = self.sess.run(self.hypothesis, feed_dict=feed)
        return prediction

    def run_train(self, train_epoch: int, home_x_train, away_x_train, y_train, keep_prob=0.8, print_num=100):
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(train_epoch):
            c, _ = self.train(home_x_train, away_x_train, y_train, keep_prob)
            acc = self.get_accuracy(home_x_train, away_x_train, y_train)
            if epoch % print_num == 0:
                print("Epoch: {}, Cost: {}, Accuracy: {}".format(epoch, c, acc))
    # endregion [NN]


class Preprocessor:
    pass


def main(_):
    pre_processor = Preprocessor()
    model = Model('winner_predict_model')
    model.learning_rate = 0.01
    model.sess = tf.Session()
    model.builder(input_size='', output_size='', model_name='model_builder')
    model.run_train(
        train_epoch=1000,
        home_x_train='',
        away_x_train='',
        y_train='',
        keep_prob=0.7,
        print_num=100
    )


if __name__ == "__main__":
    tf.app.run(main)
