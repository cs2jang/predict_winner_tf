import tensorflow as tf
from model import Model
from preprocessor import Preprocessor


def main(_):
    pre_processor = Preprocessor()
    pre_processor.set_train_test_data(0.8)

    model = Model('winner_predict_model')
    model.learning_rate = 0.01
    model.sess = tf.Session()
    model.builder(
        input_size=pre_processor.input_size,
        output_size=pre_processor.output_size,
        model_name='model_builder'
    )
    model.run_train(
        train_epoch=1000,
        train_x_home=pre_processor.train_x_home,
        train_x_away=pre_processor.train_x_away,
        train_y=pre_processor.train_y,
        keep_prob=0.8,
        print_num=100
    )
    model.closer()


if __name__ == "__main__":
    tf.app.run(main)
