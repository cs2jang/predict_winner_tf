import chardet
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    def __init__(self):
        self.away_player_data = None
        self.home_player_data = None
        self.home_team_data = None
        self.away_team_data = None
        self.scores = None
        self.df_away_player_data = None
        self.df_home_player_data = None
        self.train_x_home_player = None
        self.train_x_away_player = None
        self.train_x_home_team = None
        self.train_x_away_team = None
        self.train_y = None
        self.test_x_home_player = None
        self.test_x_away_player = None
        self.test_x_home_team = None
        self.test_x_away_team = None
        self.test_y = None
        self._player_input_size = None
        self._team_input_size = None
        self._output_size = None
        self.open_file()

    # region [Function]
    @staticmethod
    def get_hra(data, tb):
        _tb = 'top_hra' if tb == 'T' else 'bottom_hra'
        _col = "{}0".format(_tb)
        _data = data.loc[:, ['GMKEY', 'hitter_hra']]
        new_data = (_data.groupby('GMKEY')
                    .apply(lambda x: [len(x)] + x.hitter_hra.tolist())
                    .apply(pd.Series)
                    .add_prefix(_tb)
                    .reset_index()).drop([_col], axis=1)
        return new_data

    # endregion [Function]

    # region [Params]
    @property
    def get_df_away_data(self):
        return self.df_away_player_data

    @property
    def get_df_home_data(self):
        return self.df_home_player_data

    @property
    def player_input_size(self):
        return self._player_input_size

    @property
    def team_input_size(self):
        return self._team_input_size

    @property
    def output_size(self):
        return self._output_size
    # endregion [Params]

    def open_file(self):
        with open('data/top_hitter_hra.csv', 'rb') as f:
            f_result = chardet.detect(f.read())  # or readline if the file is large
            df_top_hitter_hra = pd.read_csv('data/top_hitter_hra.csv', encoding=f_result['encoding'])
            df_bottom_hitter_hra = pd.read_csv('data/bottom_hitter_hra.csv', encoding=f_result['encoding'])
            df_pitchers = pd.read_csv('data/pitchers_era.csv', encoding=f_result['encoding'])
            df_scores = pd.read_csv('data/score.csv', encoding=f_result['encoding'])
            df_team_rank = pd.read_csv('data/team_rank.csv', encoding=f_result['encoding'])

        home_team = []
        away_team = []
        for i, row in df_scores.iterrows():
            away_selected = df_team_rank[(df_team_rank['date'] < row['GDAY']) & (
                        df_team_rank['TEAM'] == row['GMKEY'][8:10])].sort_values(by='date', ascending=False).iloc[0, 2:]
            home_selected = df_team_rank[(df_team_rank['date'] < row['GDAY']) & (
                        df_team_rank['TEAM'] == row['GMKEY'][10:12])].sort_values(by='date', ascending=False).iloc[0, 2:]
            away_team.append(away_selected.tolist())
            home_team.append(home_selected.tolist())
        self.home_team_data = np.array(home_team)
        self.away_team_data = np.array(away_team)

        df_t_hra = self.get_hra(df_top_hitter_hra, 'T')
        df_b_hra = self.get_hra(df_bottom_hitter_hra, 'B')

        df_t_pitchers = df_pitchers[df_pitchers['TB'] == 'T'].drop(['TB', 'PCODE', 'NAME'], axis=1)
        df_b_pitchers = df_pitchers[df_pitchers['TB'] == 'B'].drop(['TB', 'PCODE', 'NAME'], axis=1)

        df_t_pitcher = df_t_pitchers.rename(columns={"pitcher_era": "top_pitcher_era"})
        df_b_pitcher = df_b_pitchers.rename(columns={"pitcher_era": "bottom_pitcher_era"})

        self.df_away_player_data = pd.merge(df_t_hra, df_t_pitcher, on='GMKEY').drop(['GMKEY'], axis=1)
        self.df_home_player_data = pd.merge(df_b_hra, df_b_pitcher, on='GMKEY').drop(['GMKEY'], axis=1)

        self.scores = df_scores[['TPOINT', 'BPOINT']].as_matrix()
        # define scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        # make int32 for one hot encode
        self.away_player_data = scaler.fit_transform(self.df_away_player_data)
        self.home_player_data = scaler.fit_transform(self.df_home_player_data)

    def set_train_test_data(self, rate):
        data_size = self.away_player_data.shape[0]
        train_length = int(data_size * rate)
        train_indices = np.random.choice(data_size, train_length)
        test_indices = [i for i in range(data_size) if i not in train_indices]

        self.train_x_home_player = [self.home_player_data[i] for i in train_indices]
        self.train_x_away_player = [self.away_player_data[i] for i in train_indices]
        self.train_x_home_team = [self.home_team_data[i] for i in train_indices]
        self.train_x_away_team = [self.away_team_data[i] for i in train_indices]
        self.train_y = [self.scores[i] for i in train_indices]

        self.test_x_home_player = [self.home_player_data[i] for i in test_indices]
        self.test_x_away_player = [self.away_player_data[i] for i in test_indices]
        self.test_x_home_team = [self.home_team_data[i] for i in test_indices]
        self.test_x_away_team = [self.away_team_data[i] for i in test_indices]
        self.test_y = [self.scores[i] for i in test_indices]

        self._player_input_size = self.away_player_data.shape[1]
        self._team_input_size = self.away_team_data.shape[1]
        self._output_size = self.scores.shape[1]
