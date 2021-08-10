import itertools
import pprint
import json
import random

import xgboost
from alpha.strategies.databunch import DataBunch
import os
import pickle
from typing import Callable, List, Tuple, Union
from ruamel.yaml import YAML
from numpy.typing import ArrayLike
import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
)
from xgboost import XGBClassifier, XGBModel, XGBRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from scipy.stats import randint, uniform
from typing import NewType
import datetime
from omicron.models.securities import Securities
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
import logging

logger = logging.getLogger(__name__)

# noqa
Frame = NewType("Frame", (datetime.date, datetime.datetime))


class BaseXGBoostStrategy:
    """
    Base class for XGBoost strategies.
    """

    def __init__(
        self,
        name: str,
        data_home: str,
        base_model="regressor",
        eval_metric: Union[str, Callable] = None,
        early_stopping_rounds=5,
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
        self._version = None

        self.base_model = base_model
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds

        # it's loaded when created from `load`
        self.model: XGBModel = None
        self.data_home = os.path.join(data_home, name.lower())

        os.makedirs(self.data_home, exist_ok=True)

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value: str):
        """
        set the version of the model
        """
        self._version = value

    @classmethod
    def load(cls, name: str, version: str, datahome: str):
        """load model and its meta from `path`"""

        name = name.lower()
        model = os.path.join(datahome, f"{name}.{version}.model.pkl")
        with open(model, "rb") as f:
            model = pickle.load(f)

        meta_file = os.path.join(datahome, f"{name}.{version}.meta.json")
        with open(meta_file, "r") as f:
            meta = json.load(f)

        params = {
            k: meta[k] for k in ["base_model", "eval_metric", "early_stopping_rounds"]
        }

        s = cls(name, **params)
        s.model = model

        return s

    def make_dataset(self, name: str, *args, **kwargs) -> DataBunch:
        """make dataset

        the dataset should be a dict with keys:

        data: numpy array of shape (n, m) where n is the number of instances and m is the number of features.
        target: numpy array of shape (n,) where n is the number of instances in data.

        it could also contains other keys, like:
        raw: the raw data
        desc: description of the dataset, for example, the name of the dataset, how the dataset was generated, etc.
        """
        raise NotImplementedError

    def save(self, model, report: str, ds: DataBunch):
        """save the model, hyper params, report and reference to the dataset

        if version is set, save the model and hyper params under the version directory

        Args:
            model: the model
            report: path to save the report
            ds: the dataset

        Returns:
            path to the saved model
        """
        path = self.data_home

        if self.version is not None:
            path = os.path.join(path, self.version)
            os.makedirs(path, exist_ok=True)

        stem = f"{self.name.lower()}.{self.version}"
        model_file = os.path.join(path, f"{stem}.model")
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        meta_file = os.path.join(path, f"{stem}.meta")

        meta = model.get_params()
        meta["base_model"] = self.base_model
        meta["eval_metric"] = self.eval_metric

        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=4)

        desc_file = os.path.join(path, f"{stem}.desc")
        with open(desc_file, "w") as f:
            f.writelines("the model is trained on the following dataset:\n")
            f.writelines(ds.desc)
            f.writelines("\n")
            f.writelines("final report as following\n")
            f.writelines(report)

    def x_transform(self, *args, **kwargs):
        """
        Transform train data into X, invoked by `load_data`. Usually get overriden by subclasses.
        """
        raise NotImplementedError

    def y_transform(self, *args, **kwargs):
        """tranform data into target, invoked by `load_data`. Usually get overriden by subclasses."""
        raise NotImplementedError

    def _fit(self, model, X, y, X_test, y_test):
        """
        Fit the model to a batch of training instances.

        Usually called by `fit`.

        Args:
            model: the model to be fit
            X (numpy.ndarray): A numpy array of shape (n, m) where n is the number of instances and m is the number of features.
            y (numpy.ndarray): A numpy array of shape (n,) where n is the number of instances in X.
            X_test (numpy.ndarray, optional): A numpy array of shape (n, m) where n is the number of instances in X_test.
            y_test (numpy.ndarray, optional): A numpy array of shape (n,) where n is the number of instances in y_test.
        """
        folds = RepeatedStratifiedKFold(n_splits=5, random_state=78)

        i = 1
        for train_index, valid_index in folds.split(X, y):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric=self.eval_metric,
                early_stopping_rounds=self.early_stopping_rounds,
            )

            print(f"{i}/{folds.get_n_splits()} iterations: {model.best_score}")

        preds = model.predict(X_test)
        if self.base_model == "regressor":
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            std = np.std(y_test - preds)
            report = f"rmse: {rmse}, std: {std}"
        else:
            report = classification_report(y_test, preds)

        print("final result")
        print(report)

        return model, report

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

    def fit(self, ds: DataBunch, params=None, scoring=None):
        """
        use grid search to find the best parameters and finally tune the model, save the model and report

        :param params: A dictionary of parameters.
        :return: self
        """
        params = params or {
            "colsample_bytree": uniform(0.7, 0.3),
            "gamma": uniform(0, 0.5),
            "learning_rate": uniform(0.03, 1),
            "max_depth": randint(2, 6),
            "n_estimators": randint(100, 150),
            "subsample": uniform(0.6, 0.4),
        }

        if self.base_model == "regressor":
            model, report = self._grid_search_on_regressor(ds, params, scoring)
        else:
            model, report = self._grid_search_on_classifier(ds, params, scoring)

        self.save(model, report, ds)

    def _grid_search_on_regressor(self, ds: DataBunch, params, scoring=None):
        model = XGBRegressor()

        search = RandomizedSearchCV(
            model,
            param_distributions=params,
            random_state=78,
            n_iter=200,  # the number of params to try
            cv=5,
            verbose=1,
            n_jobs=1,
            return_train_score=True,
            scoring=scoring,
            refit=True,
        )

        ds.train_test_split(stratified=False)
        search.fit(ds.X_train, ds.y_train)

        report = self._report_best_scores(search.cv_results_)
        model = search.best_estimator_

        report += "\n Final test result sample(first 20)\n"
        report += "True\tPrediction\n"
        preds = model.predict(ds.X_test)
        print("True\tPrediction")
        for i in range(20):
            console_output = f"{ds.y_test[i] * 100:.1f}\t{preds[i] * 100:.1f}"
            print(console_output)
            report += f"{console_output}\n"

        return search.best_estimator_, report


    def _grid_search_on_classifier(self, ds: DataBunch, params, scoring=None):
        model = XGBClassifier()

        search = RandomizedSearchCV(
            model,
            param_distributions=params,
            random_state=78,
            n_iter=100,
            cv=10,
            verbose=0,
            n_jobs=1,
            return_train_score=True,
            scoring=scoring,
            refit=True,  # do the refit oursel
        )

        ds.train_test_split()

        fit_params = {
            "eval_set": [(ds.X_test, ds.y_test)],
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        search.fit(ds.X_train, ds.y_train, **fit_params)

        model = search.best_estimator_

        preds = model.predict(ds.X_test)
        report = classification_report(ds.y_test, preds)

        print("final result")
        print(report)

        return model, report

    def _report_best_scores(self, results, n_top=3) -> str:
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

    def dataset_scope(self, start: Frame, end: Frame) -> List[Tuple[str, Frame]]:
        """generate sample points for making dataset.

        Use this function to exclude duplicate points.

        Args:
            start: start frame
            end: end frame

        Returns:
            A list of (frame, value)
        """
        secs = Securities()
        codes = secs.choose(_types=["stock"])
        frames = [tf.int2date(x) for x in tf.get_frames(start, end, FrameType.DAY)]

        permutations = list(itertools.product(codes, frames))

        logger.info("%s permutaions in total", len(codes) * len(frames))
        return random.sample(permutations, len(codes) * len(frames))
