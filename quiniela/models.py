import pickle
import numpy as np
import json
from sklearn import neighbors
import warnings
import sys
warnings.filterwarnings("ignore")


class QuinielaModel:
    """ This model is based on a Kneighbors classification with the following configuration:
            - n_neighbors = 7.
            - weights = "uniform".
            - algorithm = "ball_tree".

        It uses 4 features:
            - Average abolute ranking in the training seasons. If a team is in season 2,
                its absolute ranking is considered, adding the teams in season 1.
            - Average net goals (scored - conceded) in the training seasons.
            - Home team, encoded as a number.
            - Away team, encoded as a number.

        The features calculated in the training step (av. ranking and
            net goals) are then considered in the prediction step
            If a team is not found in the training step, it is considered
            to be of a low division and is assigned bad scores.
            This is a good approach if enough training seasons are considered
            (at least 5), with good results when training with 10 seasons (2010-2020).
        
        These features are stored in the file "train_features.json".
    """
    def __init__(self):
        n_neighbors = 7
        weights = "uniform"
        algorithm = "ball_tree"
        self.model = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, algorithm=algorithm)

    def train(self, train_data):
        train_df = train_data.copy(deep=True)
        train_df = train_df.dropna(subset="score")
        train_df[["home_goals", "away_goals"]] = train_df["score"].str.split(":", expand=True).astype(float)
        train_df["goal diff"] = (train_df["home_goals"] - train_df["away_goals"])
        train_df["results"] = np.where(train_df["goal diff"] > 0, "1", np.where(train_df["goal diff"] < 0, "2", "X"))
        dic_import_data = {}
        with open("feature_data.json", "r") as infile:
            dic_import_data = json.load(infile)

        dic_feature_data = {}

        train_df["rank_diff"] = 0
        train_df["goal_diff"] = 0

        st_index = train_df.index[0]
        end_index = train_df.index[-1]
        total_elems = end_index - st_index
        curr_progress = 0

        for index, row in train_df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            if home_team not in dic_feature_data:
                avg_ranking_goals(home_team, train_df, dic_import_data, dic_feature_data)
            if away_team not in dic_feature_data:
                avg_ranking_goals(away_team, train_df, dic_import_data, dic_feature_data)
            train_df.loc[row.name, "rank_diff"] = dic_feature_data[home_team][0] - dic_feature_data[away_team][0]
            train_df.loc[row.name, "goal_diff"] = dic_feature_data[home_team][1] - dic_feature_data[away_team][1]
            train_df.loc[row.name, "home_team"] = dic_import_data["teams_codif"][home_team]
            train_df.loc[row.name, "away_team"] = dic_import_data["teams_codif"][away_team]
            update_progress(float(curr_progress/total_elems))
            curr_progress += 1

        with open("train_features.json", "w") as outfile:
            json.dump(dic_feature_data, outfile)
        print("Data stored in 'train_features.json' file.")
        col_names = ["rank_diff", "goal_diff", "home_team", "away_team"]
        x_train = train_df[col_names]
        y_train = train_df.results
        self.model.fit(x_train, y_train)

    def predict(self, predict_data):
        test_df = predict_data.copy(deep=True)
        with open("train_features.json", "r") as infile:
            dic_feature_data = json.load(infile)
        with open("feature_data.json", "r") as infile:
            dic_import_data = json.load(infile)

        for index, row in test_df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            if home_team not in dic_feature_data:
                avg_rank_home = 30
                avg_goals_home = -10
            else:
                avg_rank_home = dic_feature_data[home_team][0]
                avg_goals_home = dic_feature_data[home_team][1]

            if away_team not in dic_feature_data:
                avg_rank_away = 30
                avg_goals_away = -10
            else:
                avg_rank_away = dic_feature_data[away_team][0]
                avg_goals_away = dic_feature_data[away_team][1]

            test_df.loc[row.name, "rank_diff"] = avg_rank_home - avg_rank_away
            test_df.loc[row.name, "goal_diff"] = avg_goals_home - avg_goals_away
            test_df.loc[row.name, "home_team"] = dic_import_data["teams_codif"][home_team]
            test_df.loc[row.name, "away_team"] = dic_import_data["teams_codif"][away_team]

        col_names = ["rank_diff", "goal_diff", "home_team", "away_team"]
        x_test = test_df[col_names]
        pred_results = self.model.predict(x_test)
        return pred_results

    @classmethod
    def load(cls, filename):
        """ Load model from file """
        with open(filename, "rb") as f:
            model = pickle.load(f)
            assert type(model) == cls
        print(type(model))
        return model

    def save(self, filename):
        """ Save a model in a file """
        with open(filename, "wb") as f:
            pickle.dump(self, f)


def avg_ranking_goals(team, df, gen_dic, res_dic):
    """ Function used to obtain the average ranking and goals for every team.
    This data is then used in the prediction.

    Args:
        team (string): name of the team to compute the averages
        df (DataFrame): training dataframe
        gen_dic (dictionary): general information dictionary containing data
            about the ranking and goals per season
        res_dic (dictionary): dictionary where results are stored
    """
    ranking = []
    net_goals = []
    seasons = df["season"].unique()
    for season in seasons:
        try:
            ranking.append(gen_dic["seasonal_data"][season][team][0])
        except KeyError:
            pass
        try:
            net_goals.append(gen_dic["seasonal_data"][season][team][1])
        except KeyError:
            pass
    avg_ranking = round(np.average(ranking), 3)
    avg_net_goals = round(np.average(net_goals), 3)
    res_dic[team] = [avg_ranking, avg_net_goals]


def update_progress(progress):
    barLength = 30  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Training the model...\r\n"
    block = int(round(barLength*progress))
    text = "\rGenerating features: [{0}] {1}% {2}".format("█"*block + "-"*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()
