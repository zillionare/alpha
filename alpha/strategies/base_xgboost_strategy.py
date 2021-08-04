

from math import fabs
import os
import pickle
from ruamel.yaml import YAML
from numpy.typing import ArrayLike
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor


class BaseXGBoostStrategy:
    """
    Base class for XGBoost strategies.
    """
    def __init__(self, name:str, params:dict, base_model='regressor'):
        self.name = name
        self.params = params
        self.base_model = base_model
        self.model = None

        self.data_train = None
        self.data_valid = None
        self.data_test = None

    @classmethod
    def load(cls, path:str, name:str):
        """load model and its meta from `path`"""

        model = os.path.join(path, f"{name}_model.pkl")
        with open(model, 'rb') as f:
            model = pickle.load(f)

        meta = os.path.join(path, f"{name}_desc.yml")
        with open(meta, "rb") as f:
            yaml = YAML()
            yaml.default_flow_style = False
            params = yaml.load(f)

        s = cls(name, params)
        s.model = model

        return s


    def save(self, path:str):
        """save model and its meta info to `path`"""
        model = os.path.join(path, f"{self.name}_model.pkl")
        with open(model, 'wb') as f:
            pickle.dump(self.model, f)

        meta = os.path.join(path, f"{self.name}_desc.yml")
        with open(meta, "wb") as f:
            yaml = YAML()
            yaml.default_flow_style = False
            yaml.dump(self.params, f)

    def transform(self, X):
        """
        Transform a batch of instances.

        :param X: A numpy array of shape (n, m) where n is the number of instances and m is the number of features.
        :return: A numpy array of shape (n, m) where n is the number of instances in X and m is the number of features.
        """
        raise NotImplementedError

    def fit(self, X, y):
        """
        Fit the model to a batch of training instances.

        :param X: A numpy array of shape (n, m) where n is the number of instances and m is the number of features.
        :param y: A numpy array of shape (n,) where n is the number of instances in X.
        :return: self
        """
        return self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the class of a batch of instances.

        :param X: A numpy array of shape (n, m) where n is the number of instances and m is the number of features.
        :return: A numpy array of shape (n,) where n is the number of instances in X.
        """
        X_ = self.transform(X)
        return self.model.predict(X_)

    def shuffle(self, data: ArrayLike, seed:int=78)->ArrayLike:
        """shuffle the `data`"""
        np.random.seed(seed)
        indice = np.random.choice(len(data), size=len(data), replace=False)

        return data[indice]

    def load_data(self, data_file:str, shuffle:bool=True, seed:int=78, valid_pct:float=0.2, test_pct:float=0.1):
        """load data from `data_file`, then do the shuffle, split and return train, valid and test set

        Args:
            data_file: path to the data file
            shuffle: whether to shuffle the data
            seed: random seed
            valid_pct: percentage of validation set
            test_pct: percentage of test set
       """
        data = np.load(data_file)
        if shuffle:
            data = self.shuffle(data, seed)

        n = len(data)
        valid_n = int(valid_pct * n)
        test_n = int(test_pct * n)
        train_n = n - valid_n - test_n

        train = data[:train_n]
        valid = data[train_n:train_n+valid_n]
        test = data[train_n+valid_n:]

        self.data_train = train
        self.data_valid = valid
        self.data_test = test

        return train, valid, test

    def get_X(self, dataset:str):
        """get train data from dataset"""
        raise NotImplementedError

    def get_y(self, dataset:str):
        """get target from dataset"""
        raise NotImplementedError
        
    def grid_search(self, params, scoring=None):
        """
        Grid search the best parameters.

        :param params: A dictionary of parameters.
        :return: self
        """
        if self.base_model == 'regressor':
            model = XGBRegressor()
        else:
            model = XGBClassifier()

        search = RandomizedSearchCV(
            model,
            param_distributions=params,
            random_state=78,
            n_iter=200,
            cv=3,
            verbose=1,
            n_jobs=-1,
            return_train_score=True,
            scoring=scoring
        )

        X = self.get_X(dataset='train')
        y = self.get_y(dataset='train')

        search.fit(X, y)

        self._report_best_params(search.best_params_)

    def _report_best_scores(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results["rank_test_score"] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print(
                    "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results["mean_test_score"][candidate],
                        results["std_test_score"][candidate],
                    )
                )
                print("Parameters: {0}".format(results["params"][candidate]))
                print("")
