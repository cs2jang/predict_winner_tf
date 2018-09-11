import tensorflow as tf
import numpy as np
import pandas as pd


class Model:
    def __init__(self, name):
        self.name = name
        self.xavier = tf.contrib.layers.xavier_initializer()
        self._learning_rate = 0.01
        self._training_epoch = 1000

    # region [Functions]
    def get_weight(self, name, w_size):
        return tf.get_variable(name, w_size, initializer=self.xavier)

    def get_bias(self, name, b_size):
        return tf.Variable(tf.random_normal(b_size), name=name)

    def get_layer(self, act, w_in, w_out, b, drop=None):
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
    # endregion [Hyper Params]

    def builder(self, input_size, output_size, layer_name, act=tf.nn.relu):
        with tf.variable_scope(layer_name):
            X_home = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
            X_away = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
            Y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])
            keep_prob = tf.placeholder(dtype=tf.float32)

            n_merged = input_size * 2 * 2
            n_nuerons_1 = input_size * 2 * 2 * 10

            weights = {
                'w1_home': self.get_weight('W1_home_hidden', [input_size, input_size * 2]),
                'w2_home': self.get_weight('W2_home_hidden', [input_size * 2, input_size]),
                'w3_home': self.get_weight('W3_home_hidden', [input_size, input_size * 2]),
                'w1_away': self.get_weight('W1_away_hidden', [input_size, input_size * 2]),
                'w2_away': self.get_weight('W2_away_hidden', [input_size * 2, input_size]),
                'w3_away': self.get_weight('W3_away_hidden', [input_size, input_size * 2]),
                'w1': self.get_weight('W1_hidden', [n_merged, n_nuerons_1]),
                'w2': self.get_weight('W2_hidden', [n_nuerons_1, n_nuerons_1]),
                'w3': self.get_weight('W3_hidden', [n_nuerons_1, n_merged]),
                'w4': self.get_weight('W4_hidden', [n_merged, output_size]),
            }

            bias = {
                'b1_home': self.get_bias('b1_home', [input_size * 2]),
                'b2_home': self.get_bias('b2_home', [input_size]),
                'b3_home': self.get_bias('b3_home', [input_size * 2]),
                'b1_away': self.get_bias('b1_away', [input_size * 2]),
                'b2_away': self.get_bias('b2_away', [input_size]),
                'b3_away': self.get_bias('b3_away', [input_size * 2]),
                'b1': self.get_bias('b1', [n_nuerons_1]),
                'b2': self.get_bias('b2', [n_nuerons_1]),
                'b3': self.get_bias('b3', [n_merged]),
                'b4': self.get_bias('b4', [output_size]),
            }

            h_l_1 = self.get_layer(act, X_home, weights['w1_home'], bias['b1_home'], keep_prob)
            h_l_2 = self.get_layer(act, h_l_1, weights['w2_home'], bias['b2_home'], keep_prob)
            h_l_3 = self.get_layer(act, h_l_2, weights['w3_home'], bias['b3_home'], keep_prob)

            a_l_1 = self.get_layer(act, X_away, weights['w1_away'], bias['b1_away'], keep_prob)
            a_l_2 = self.get_layer(act, a_l_1, weights['w2_away'], bias['b2_away'], keep_prob)
            a_l_3 = self.get_layer(act, a_l_2, weights['w3_away'], bias['b3_away'], keep_prob)

            merge_layers = tf.concat([a_l_3, h_l_3], 1)

            l_1 = self.get_layer(act, merge_layers, weights['w1'], bias['b1'], keep_prob)
            l_2 = self.get_layer(act, l_1, weights['w2'], bias['b2'], keep_prob)
            l_3 = self.get_layer(act, l_2, weights['w3'], bias['b3'], keep_prob)

            self.hypothesis = tf.add(tf.matmul(l_3, weights['w4']), bias['b4'])

        cost = tf.reduce_mean(tf.square(self.hypothesis - Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(cost)