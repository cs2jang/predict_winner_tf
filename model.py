import tensorflow as tf


class Model:
    def __init__(self, name):
        tf.set_random_seed(777)
        self.name = name
        self.xavier = tf.contrib.layers.xavier_initializer()
        self._learning_rate = 0.01
        self._training_epoch = 1000
        self._sess = None
        self.train_writer = None
        self.hypothesis = None
        self.X_home_player = None
        self.X_away_player = None
        self.X_home_team = None
        self.X_away_team = None
        self.Y = None
        self.keep_prob = None
        self.cost = None
        self.optimizer = None
        self.log_dir = "./logs/"
        self.merged = None
        self.train_writer = None
        self.test_writer = None
        self.accuracy = None

    # region [Functions]
    def get_weight(self, name, w_size):
        return tf.get_variable(name, w_size, initializer=self.xavier)

    @staticmethod
    def get_bias(name, b_size):
        return tf.Variable(tf.random_normal(b_size), name=name)

    @staticmethod
    def get_layer(act, w_in, w_out, b, drop=None):
        layer = act(tf.add(tf.matmul(w_in, w_out), b))
        if drop is not None:
            layer = tf.nn.dropout(layer, keep_prob=drop)
        return layer

    def closer(self):
        self.train_writer.close()
        self.test_writer.close()
        self.sess.close()
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

    def set_summary(self):
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.log_dir + '/test')
    # endregion [TF Board]

    # region [NN]
    def builder(self, player_input_size, team_input_size, output_size, model_name, act=tf.nn.relu):
        with tf.name_scope('input'):
            self.X_home_player = tf.placeholder(dtype=tf.float32, shape=[None, player_input_size])
            self.X_away_player = tf.placeholder(dtype=tf.float32, shape=[None, player_input_size])
            self.X_home_team = tf.placeholder(dtype=tf.float32, shape=[None, team_input_size])
            self.X_away_team = tf.placeholder(dtype=tf.float32, shape=[None, team_input_size])
            self.Y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

        with tf.name_scope(model_name):
            with tf.name_scope('dropout'):
                self.keep_prob = tf.placeholder(dtype=tf.float32)
                tf.summary.scalar('dropout_keep_probability', self.keep_prob)

            n_merged = (player_input_size * 2 * 2) + (team_input_size * 2 * 2)
            n_neurons_1 = n_merged * 4

            with tf.name_scope('weights'):
                weights = {
                    'w1_home_player': self.get_weight('W1_home_player_hidden', [player_input_size, player_input_size * 2]),
                    'w2_home_player': self.get_weight('W2_home_player_hidden', [player_input_size * 2, player_input_size]),
                    'w3_home_player': self.get_weight('W3_home_player_hidden', [player_input_size, player_input_size * 2]),
                    'w1_away_player': self.get_weight('W1_away_player_hidden', [player_input_size, player_input_size * 2]),
                    'w2_away_player': self.get_weight('W2_away_player_hidden', [player_input_size * 2, player_input_size]),
                    'w3_away_player': self.get_weight('W3_away_player_hidden', [player_input_size, player_input_size * 2]),
                    'w1_home_team': self.get_weight('W1_home_team_hidden', [team_input_size, team_input_size * 2]),
                    'w2_home_team': self.get_weight('W2_home_team_hidden', [team_input_size * 2, team_input_size]),
                    'w3_home_team': self.get_weight('W3_home_team_hidden', [team_input_size, team_input_size * 2]),
                    'w1_away_team': self.get_weight('W1_away_team_hidden', [team_input_size, team_input_size * 2]),
                    'w2_away_team': self.get_weight('W2_away_team_hidden', [team_input_size * 2, team_input_size]),
                    'w3_away_team': self.get_weight('W3_away_team_hidden', [team_input_size, team_input_size * 2]),
                    'w1': self.get_weight('W1_hidden', [n_merged, n_neurons_1]),
                    'w2': self.get_weight('W2_hidden', [n_neurons_1, n_neurons_1]),
                    'w3': self.get_weight('W3_hidden', [n_neurons_1, n_merged]),
                    'w4': self.get_weight('W4_hidden', [n_merged, output_size]),
                }
                for i, w in weights.items():
                    self.variable_summaries(w)

            with tf.name_scope('biases'):
                biases = {
                    'b1_home_player': self.get_bias('b1_home_player', [player_input_size * 2]),
                    'b2_home_player': self.get_bias('b2_home_player', [player_input_size]),
                    'b3_home_player': self.get_bias('b3_home_player', [player_input_size * 2]),
                    'b1_away_player': self.get_bias('b1_away_player', [player_input_size * 2]),
                    'b2_away_player': self.get_bias('b2_away_player', [player_input_size]),
                    'b3_away_player': self.get_bias('b3_away_player', [player_input_size * 2]),
                    'b1_home_team': self.get_bias('b1_home_team', [team_input_size * 2]),
                    'b2_home_team': self.get_bias('b2_home_team', [team_input_size]),
                    'b3_home_team': self.get_bias('b3_home_team', [team_input_size * 2]),
                    'b1_away_team': self.get_bias('b1_away_team', [team_input_size * 2]),
                    'b2_away_team': self.get_bias('b2_away_team', [team_input_size]),
                    'b3_away_team': self.get_bias('b3_away_team', [team_input_size * 2]),
                    'b1': self.get_bias('b1', [n_neurons_1]),
                    'b2': self.get_bias('b2', [n_neurons_1]),
                    'b3': self.get_bias('b3', [n_merged]),
                    'b4': self.get_bias('b4', [output_size]),
                }
                # for i, bias in biases.items():
                #     self.variable_summaries(bias)

            with tf.name_scope('layers'):
                h_p_l_1 = self.get_layer(act, self.X_home_player, weights['w1_home_player'], biases['b1_home_player'], self.keep_prob)
                h_p_l_2 = self.get_layer(act, h_p_l_1, weights['w2_home_player'], biases['b2_home_player'], self.keep_prob)
                h_p_l_3 = self.get_layer(act, h_p_l_2, weights['w3_home_player'], biases['b3_home_player'], self.keep_prob)
                tf.summary.histogram('home_player_1_activation', h_p_l_1)
                tf.summary.histogram('home_player_2_activation', h_p_l_2)
                tf.summary.histogram('home_player_3_activation', h_p_l_3)

                a_p_l_1 = self.get_layer(act, self.X_away_player, weights['w1_away_player'], biases['b1_away_player'], self.keep_prob)
                a_p_l_2 = self.get_layer(act, a_p_l_1, weights['w2_away_player'], biases['b2_away_player'], self.keep_prob)
                a_p_l_3 = self.get_layer(act, a_p_l_2, weights['w3_away_player'], biases['b3_away_player'], self.keep_prob)
                tf.summary.histogram('away_player_1_activation', a_p_l_1)
                tf.summary.histogram('away_player_2_activation', a_p_l_2)
                tf.summary.histogram('away_player_3_activation', a_p_l_3)

                h_t_l_1 = self.get_layer(act, self.X_home_team, weights['w1_home_team'], biases['b1_home_team'], self.keep_prob)
                h_t_l_2 = self.get_layer(act, h_t_l_1, weights['w2_home_team'], biases['b2_home_team'], self.keep_prob)
                h_t_l_3 = self.get_layer(act, h_t_l_2, weights['w3_home_team'], biases['b3_home_team'], self.keep_prob)
                tf.summary.histogram('home_team_1_activation', h_t_l_1)
                tf.summary.histogram('home_team_2_activation', h_t_l_2)
                tf.summary.histogram('home_team_3_activation', h_t_l_3)

                a_t_l_1 = self.get_layer(act, self.X_away_team, weights['w1_away_team'], biases['b1_away_team'], self.keep_prob)
                a_t_l_2 = self.get_layer(act, a_t_l_1, weights['w2_away_team'], biases['b2_away_team'], self.keep_prob)
                a_t_l_3 = self.get_layer(act, a_t_l_2, weights['w3_away_team'], biases['b3_away_team'], self.keep_prob)
                tf.summary.histogram('away_team_1_activation', a_t_l_1)
                tf.summary.histogram('away_team_2_activation', a_t_l_2)
                tf.summary.histogram('away_team_3_activation', a_t_l_3)

                merge_layers = tf.concat([a_p_l_3, h_p_l_3, a_t_l_3, h_t_l_3], 1)

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
            self.cost = tf.reduce_mean(tf.square(tf.subtract(self.hypothesis, self.Y)))
            tf.summary.scalar('cost', self.cost)
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self.cost)

    def get_accuracy(self, home_team_x_data, away_team_x_data, home_player_x_data, away_player_x_data, data_y, keep_prob=1.0):
        data_feed = {
            self.X_home_player: home_player_x_data,
            self.X_away_player: away_player_x_data,
            self.X_home_team: home_team_x_data,
            self.X_away_team: away_team_x_data,
            self.keep_prob: keep_prob
        }
        predict = self.sess.run(self.hypothesis, feed_dict=data_feed)

        prediction = tf.map_fn(
            lambda x: x[0] > x[1],
            predict,
            dtype=bool
        )

        answer = tf.map_fn(
            lambda x: x[0] > x[1],
            data_y,
            dtype=bool
        )

        accuracy = tf.divide(tf.reduce_sum(tf.cast(tf.equal(prediction, answer), dtype=tf.int32)), len(data_y))
        return self.sess.run(accuracy)

    def train(self, train_x_home_team, train_x_away_team, train_x_home_player, train_x_away_player, train_y, keep_prob):
        train_feed = {
            self.X_home_player: train_x_home_player,
            self.X_away_player: train_x_away_player,
            self.X_home_team: train_x_home_team,
            self.X_away_team: train_x_away_team,
            self.Y: train_y,
            self.keep_prob: keep_prob
        }
        summary, _cost, _opt = self.sess.run([self.merged, self.cost, self.optimizer], feed_dict=train_feed)
        return summary, _cost, _opt

    def test(self, test_x_home, test_x_away, keep_prob=1.0):
        test_feed = {self.X_home_player: test_x_home, self.X_away_player: test_x_away, self.keep_prob: keep_prob}
        prediction = self.sess.run(self.hypothesis, feed_dict=test_feed)
        return prediction

    def predict(self, home_x_data, away_x_data, keep_prob=1.0):
        feed = {self.X_home_player: home_x_data, self.X_away_player: away_x_data, self.keep_prob: keep_prob}
        prediction = self.sess.run(self.hypothesis, feed_dict=feed)
        return prediction

    def run_train(self, train_epoch: int, train_x_home_team, train_x_away_team, train_x_home_player, train_x_away_player, train_y, keep_prob=0.8, print_num=100):
        self.set_summary()
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(train_epoch):
            summary, c, _ = self.train(train_x_home_team, train_x_away_team, train_x_home_player, train_x_away_player, train_y, keep_prob)
            acc = self.get_accuracy(train_x_home_team, train_x_away_team, train_x_home_player, train_x_away_player, train_y)
            if epoch % print_num == 0:
                self.train_writer.add_summary(summary, epoch)
                print("Epoch: {}, Cost: {}, Accuracy: {}".format(epoch, c, acc))

    def run_test(self, test_x_home_team, test_x_away_team, test_x_home_player, test_x_away_player, test_y):
        accuracy = self.get_accuracy(test_x_home_team, test_x_away_team, test_x_home_player, test_x_away_player, test_y)
        print("Test data Accuracy : ", accuracy)
    # endregion [NN]
