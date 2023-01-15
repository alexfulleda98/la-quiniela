import pickle
import numpy as np
from sklearn import svm


class QuinielaModel:
    
    def train(self, train_data):
        train_data[["home_goals", "away_goals"]] = train_data["score"].str.split(":", expand=True).astype(float)
        train_data = train_data.dropna(subset="score")
        train_data["goal diff"] = (train_data["home_goals"] - train_data["away_goals"])
        train_data["results"] = np.where(train_data["goal diff"] > 0, "1", np.where(train_data["goal diff"] < 0, "2", "X"))
        train_data['home_win'] = train_data.home_goals > train_data.away_goals

        X = train_data[['home_win']]
        y = train_data.results

        clf = svm.SVC()
        clf.fit(X, y)
        return clf

    def predict(self, predict_data):
        # Do something here to predict
        predict_data[["home_goals", "away_goals"]] = predict_data["score"].str.split(":", expand=True).astype(float)
        predict_data = predict_data.dropna(subset="score")
        predict_data["goal diff"] = (predict_data["home_goals"] - predict_data["away_goals"])
        predict_data["results"] = np.where(predict_data["goal diff"] > 0, "1", np.where(predict_data["goal diff"] < 0, "2", "X"))
        predict_data['home_win'] = predict_data.home_goals > predict_data.away_goals

        X = predict_data[['home_win']]
        return self.predict(X)

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
