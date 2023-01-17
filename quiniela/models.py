import pickle
import numpy as np
import json
from sklearn import neighbors
import warnings
import sys
warnings.filterwarnings("ignore")


class QuinielaModel:

    def __init__(self):
        n_neighbors = 30
        weights = "distance"
        self.model = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, algorithm="auto")

    def train(self, train_data):
        results_dic = {}
        train_data = train_data.dropna(subset="score")
        train_data[["home_goals", "away_goals"]] = train_data["score"].str.split(":", expand=True).astype(float)
        train_data["goal diff"] = (train_data["home_goals"] - train_data["away_goals"])
        train_data["results"] = np.where(train_data["goal diff"] > 0, "1", np.where(train_data["goal diff"] < 0, "2", "X"))
        with open("abs_rankings.json", "r") as ranking_input:
            abs_rankings = json.load(ranking_input)
        x_train = train_data.copy(deep=True)
        x_train["win_diff"] = 0
        x_train["rank_diff"] = 0
        st_index = x_train.index[0]
        end_index = x_train.index[-1]
        total_elems = end_index - st_index
        curr_progress = 0
        for index, row in x_train.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            try:
                results_dic[home_team][away_team]
            except KeyError:
                dic_saver(home_team, away_team, x_train, results_dic, abs_rankings)
            # results_dic [team1][team2] = [win1-win2, rank1-rank2]
            x_train.loc[row.name, "win_diff"] = results_dic[home_team][away_team][0]
            x_train.loc[row.name, "rank_diff"] = results_dic[home_team][away_team][1]
            update_progress(float(curr_progress/total_elems))
            curr_progress += 1
        with open("train_feature_results.json", "w") as outfile:
            json.dump(results_dic, outfile)

        col_names = ["win_diff", "rank_diff"]
        X = x_train[col_names]
        y = x_train.results

        self.model.fit(X, y)

    def predict(self, predict_data):
        with open("train_feature_results.json", "r") as result_input:
            results_dic = json.load(result_input)
        mod_predict_data = predict_data.copy(deep=True)
        mod_predict_data["win_diff"] = 0
        mod_predict_data["rank_diff"] = 0
        print("progress : 'Generating Features ...'")
        for index, row in mod_predict_data.iterrows():
            try:
                row["home_team"]
            except KeyError:
                continue
            home_team = row["home_team"]
            away_team = row["away_team"]
            try:
                mod_predict_data.loc[row.name, "win_diff"] = results_dic[home_team][away_team][0]
                mod_predict_data.loc[row.name, "rank_diff"] = results_dic[home_team][away_team][1]
            except KeyError:
                mod_predict_data.loc[row.name, "win_diff"] = 2
                mod_predict_data.loc[row.name, "rank_diff"] = 0


        col_names = ["win_diff", "rank_diff"]
        X = mod_predict_data[col_names]
        pred_results = self.model.predict(X)
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


def inv_conv(lst):
    return [-i for i in lst]


def direct_confrontations_and_ranking(Team1, Team2, df, abs_rankings):
    ranking_team1 = []
    ranking_team2 = []
    df3 = df.loc[((df["home_team"] == Team2) & (df["away_team"] == Team1)) | ((df["home_team"] == Team1) & (df["away_team"] == Team2))]
    df3["Winner"] = np.where(df3["goal diff"] > 0, df3.home_team.values, np.where(df3["goal diff"] < 0, df3.away_team.values, "tie"))
    direct_confrontation_count = df3["Winner"].value_counts()
    if Team1 not in direct_confrontation_count and Team2 not in direct_confrontation_count:
        team1_scoring = 0
        team2_scoring = 0
    elif Team1 not in direct_confrontation_count or Team2 not in direct_confrontation_count:
        if Team1 not in direct_confrontation_count:
            team1_scoring = 0
            team2_scoring = direct_confrontation_count[Team2]
        else:
            team1_scoring = direct_confrontation_count[Team1]
            team2_scoring = 0
    else:
        team1_scoring = direct_confrontation_count[Team1]
        team2_scoring = direct_confrontation_count[Team2]

    seasons = df3["season"].unique()
    for season in seasons:
        try:
            ranking_team1.append(abs_rankings[season][Team1])
        except KeyError:
            pass
        try:
            ranking_team2.append(abs_rankings[season][Team2])
        except KeyError:
            pass
    avg_ranking_team1 = round(np.average(ranking_team1), 3)
    avg_ranking_team2 = round(np.average(ranking_team2), 3)
    return [float(team1_scoring - team2_scoring), float(avg_ranking_team1 - avg_ranking_team2)]


def dic_saver(team1, team2, df, results_dic, abs_rankings):
    subdic_direct = {}
    subdic_inv = {}
    direct_conf_res = direct_confrontations_and_ranking(team1, team2, df, abs_rankings)
    subdic_direct[team2] = direct_conf_res
    subdic_inv[team1] = inv_conv(direct_conf_res)
    results_dic[team1] = subdic_direct
    results_dic[team2] = subdic_inv


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
    text = "\rProgress: [{0}] {1}% {2}".format("█"*block + "-"*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()
