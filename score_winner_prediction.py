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
        team_input_size=pre_processor.team_input_size,
        player_input_size=pre_processor.player_input_size,
        output_size=pre_processor.output_size,
        model_name='model_builder'
    )
    model.run_train(
        train_epoch=5000,
        train_x_home_team=pre_processor.train_x_home_team,
        train_x_away_team=pre_processor.train_x_away_team,
        train_x_home_player=pre_processor.train_x_home_player,
        train_x_away_player=pre_processor.train_x_away_player,
        train_y=pre_processor.train_y,
        keep_prob=0.7,
        print_num=500
    )
    model.run_test(
        test_x_home_team=pre_processor.test_x_home_team,
        test_x_away_team=pre_processor.test_x_away_team,
        test_x_home_player=pre_processor.test_x_home_player,
        test_x_away_player=pre_processor.test_x_away_player,
        test_y=pre_processor.test_y
    )
    model.closer()


if __name__ == "__main__":
    tf.app.run(main)
