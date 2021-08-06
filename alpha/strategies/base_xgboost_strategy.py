from math import fabs
import os
import pickle
from tkinter import Y
from typing import Callable, Union
from ruamel.yaml import YAML
from numpy.typing import ArrayLike
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier, XGBModel, XGBRegressor
from sklearn.metrics import classification_report
from scipy.stats import randint, uniform

class BaseXGBoostStrategy:
    """
    Base class for XGBoost strategies.
    """

    def __init__(
        self,
        name: str,
        data_home:str,
        version: str = None,
        base_model="regressor",
        eval_metric: Union[str, Callable] = "rmse",
        early_stopping_rounds=5
    ):
        """

        Args:
            name (str): the name of the model
            data_home (str): the path to the data directory
            version (str, optional): the version of the model. Defaults to None.
            base_model (str, optional): base_model name. Defaults to "regressor".
            eval_metric (Union[str, Callable], optional): [description]. Defaults to "rmse".
            early_stopping_rounds (int, optional): [description]. Defaults to 5.
        """
        self.name = name
        self.version = version

        self.base_model = base_model
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds

        # it's loaded when created from `load`
        self.model: XGBModel = None
        self.data_home = os.path.join(data_home, name)

        os.makedirs(self.data_home, exist_ok=True)

    @classmethod
    def load(cls, path: str, name: str):
        """load model and its meta from `path`"""

        model = os.path.join(path, f"{name}_model.pkl")
        with open(model, "rb") as f:
            model = pickle.load(f)

        meta = os.path.join(path, f"{name}_desc.yml")
        with open(meta, "rb") as f:
            yaml = YAML()
            yaml.default_flow_style = False
            params = yaml.load(f)

        s = cls(name, params)
        s.model = model

        return s

    def make_dataset(self,name:str, *args, **kwargs):
        """make dataset

        the dataset should be saved in `self.data_home/self.version`/`name`.pkl

        the dataset should be a dict with keys:

        data: numpy array of shape (n, m) where n is the number of instances and m is the number of features.
        target: numpy array of shape (n,) where n is the number of instances in data.

        it could also contains other keys, like:
        raw: the raw data
        desc: description of the dataset, for example, the name of the dataset, how the dataset was generated, etc.
        """
        raise NotImplementedError

    def save(self, hyper_params: dict, report: str=None):
        """save the model, hyper params, report and reference to the dataset

        if version is set, save the model and hyper params under the version directory

        Args:
            report: path to save the report

        Returns:
            path to the saved model
        """
        path = self.data_home

        if self.version is not None:
            path = os.path.join(path, self.version)
            os.makedirs(path, exist_ok=True)

        model = os.path.join(path, f"{self.name}_model.pkl")
        with open(model, "wb") as f:
            pickle.dump(self.model, f)

        meta_file = os.path.join(path, f"{self.name}_desc.yml")

        hyper_params["report"] = report
        hyper_params["base_model"] = self.base_model
        hyper_params["eval_metric"] = self.eval_metric


        with open(meta_file, "wb") as f:
            yaml = YAML()
            yaml.default_flow_style = False
            yaml.dump(hyper_params, f)

    def x_transform(self, *args, **kwargs):
        """
        Transform train data into X, invoked by `load_data`. Usually get overriden by subclasses.
        """
        raise NotImplementedError

    def y_transform(self, *args, **kwargs):
        """tranform data into target, invoked by `load_data`. Usually get overriden by subclasses.
        """
        raise NotImplementedError

    def fit(self, X, y, X_test=None, y_test=None):
        """
        Fit the model to a batch of training instances.

        Args:
            X (numpy.ndarray): A numpy array of shape (n, m) where n is the number of instances and m is the number of features.
            y (numpy.ndarray): A numpy array of shape (n,) where n is the number of instances in X.
            X_test (numpy.ndarray, optional): A numpy array of shape (n, m) where n is the number of instances in X_test.
            y_test (numpy.ndarray, optional): A numpy array of shape (n,) where n is the number of instances in y_test.
        """
        # todo: implement early stopping
        return self.model.fit(
            X,
            y,
            early_stopping_rounds=5,
            eval_set=[(X_test, y_test)]
        )

    def predict(self, X):
        """
        Predict the class of a batch of instances.

        Args:
            X: A numpy array of shape (n, m) where n is the number of instances and m is the number of features.
        return:
            A numpy array of shape (n,) where n is the number of instances in X.
        """
        X_ = self.x_transform(X)
        return self.model.predict(X_)

    def grid_search(self, params=None, scoring=None):
        """
        Grid search the best parameters.

        :param params: A dictionary of parameters.
        :return: self
        """
        params = params or  {
            "colsample_bytree": uniform(0.7, 0.3),
            "gamma": uniform(0, 0.5),
            "learning_rate": uniform(0.03, 1),
            "max_depth": randint(2, 6),
            "n_estimators": randint(100, 150),
            "subsample": uniform(0.6, 0.4),
        }

        if self.base_model == "regressor":
            self._grid_search_on_regressor(params, scoring)
        else:
            self._grid_search_on_classifier(params, scoring)


    def _grid_search_on_regressor(self, params, scoring=None):
        model = XGBRegressor()

        search = RandomizedSearchCV(
            model,
            param_distributions=params,
            random_state=78,
            n_iter=200,
            cv=3,
            verbose=2,
            n_jobs=1,
            return_train_score=True,
            scoring=scoring,
            refit=True
        )

        X = self.get_X(dataset="train")
        y = self.get_y(dataset="train")

        search.fit(X, y)

        report = self._report_best_scores(search.best_params_)

    def _grid_search_on_classifier(self, params, scoring=None):
        model = XGBClassifier()

        search = RandomizedSearchCV(
            model,
            param_distributions=params,
            random_state=78,
            n_iter=200,
            cv=3,
            verbose=2,
            n_jobs=1,
            return_train_score=True,
            scoring=scoring,
            refit=True
        )

        X = self.get_X(dataset="train")
        y = self.get_y(dataset="train")
        search.fit(X, y)

        best_model = search.best_estimator_
        X_test = self.get_X(dataset="test")
        y_true = self.get_y(dataset="test")

        y_pred = best_model.predict(X_test)

        report = classification_report(y_true, y_pred)
        print(report)

        self.save(best_model, params, report)

    def _report_best_scores(self, results, n_top=3)->str:
        report = []
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results["rank_test_score"] == i)
            for candidate in candidates:
                report.append("Model with rank: {0}".format(i))
                report.append(
                    "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results["mean_test_score"][candidate],
                        results["std_test_score"][candidate],
                    )
                )
                report.append("Parameters: {0}".format(results["params"][candidate]))
                report.append("")

        report = "\n".join(report)
        print(report)
        return report

    async def build_dataset(self, save_to: str):
        raise NotImplementedError
