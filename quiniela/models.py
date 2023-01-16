import pickle
import numpy as np
import json


class QuinielaModel:

    def train(self, train_data):
        results_dic = {}
        train_data[["home_goals", "away_goals"]] = train_data["score"].str.split(":", expand=True).astype(float)
        train_data = train_data.dropna(subset="score")
        train_data["goal diff"] = (train_data["home_goals"] - train_data["away_goals"])
        train_data["results"] = np.where(train_data["goal diff"] > 0, "1", np.where(train_data["goal diff"] < 0, "2", "X"))        
        with open("../abs_rankings.json", "r") as ranking_input:
            abs_rankings = json.load(ranking_input)
        x_train = train_data
        x_train["win_diff"] = 0
        x_train["rank_diff"] = 0
        for index, row in x_train.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            dic_saver(home_team, away_team, x_train)
            # results_dic [team1][team2] = [win1-win2, rank1-rank2]
            x_train.at[row.name,"win_diff"] = results_dic[home_team][away_team][0]
            x_train.at[row.name,"rank_diff"] = results_dic[home_team][away_team][1]
        
        return clf

    def predict(self, predict_data):
        # Do something here to predict
        predict_data[["home_goals", "away_goals"]] = predict_data["score"].str.split(":", expand=True).astype(float)
        predict_data = predict_data.dropna(subset="score")
        predict_data["goal diff"] = (predict_data["home_goals"] - predict_data["away_goals"])
        predict_data["results"] = np.where(predict_data["goal diff"] > 0, "1", np.where(predict_data["goal diff"] < 0, "2", "X"))
        predict_data['home_win'] = predict_data.home_goals > predict_data.away_goals

        X = predict_data[['home_win']]
        return self.predict(predict_data)

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
        return [ -i for i in lst ]

    def direct_confrontations_and_ranking(Team1, Team2, df):
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
            ranking_team1.append(abs_rankings[season][Team1])
            ranking_team2.append(abs_rankings[season][Team2])
        avg_ranking_team1 = round(np.average(ranking_team1), 3)
        avg_ranking_team2 = round(np.average(ranking_team2), 3)
        return [team1_scoring - team2_scoring, avg_ranking_team1 - avg_ranking_team2]

    def dic_saver(team1, team2, df):
        subdic_direct = {}
        subdic_inv = {}
        direct_conf_res = direct_confrontations_and_ranking(team1, team2, df)
        subdic_direct[team2] = direct_conf_res
        subdic_inv[team1] = inv_conv(direct_conf_res)
        results_dic[team1] = subdic_direct
        results_dic[team2] = subdic_inv