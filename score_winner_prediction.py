import tensorflow as tf
from model import Model
from preprocessor import Preprocessor


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
