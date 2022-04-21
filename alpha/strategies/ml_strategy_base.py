import os
import pickle
from abc import ABCMeta, abstractmethod
from calendar import c
from typing import Dict

import numpy as np
from ruamel.yaml import YAML


class MLStrategyBase(metaclass=ABCMeta):
    def __init__(self, name: str, data_home: str):
        self.name = name
        self._version = None
        self.data_home = data_home
        self.model = None

        os.makedirs(self.data_home, exist_ok=True)

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value: str):
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
            meta = YAML().load(f)

        params = {
            k: meta[k] for k in ["base_model", "eval_metric", "early_stopping_rounds"]
        }

        s = cls(name, **params)
        s.model = model

        return s

    def save(self, model, report: str, ds: Dict):
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
            YAML().dump(meta, f)

        desc_file = os.path.join(path, f"{stem}.desc")
        with open(desc_file, "w") as f:
            f.writelines("the model is trained on the following dataset:\n")
            f.writelines(ds.desc)
            f.writelines("\n")
            f.writelines("final report as following\n")
            f.writelines(report)

    @classmethod
    @abstractmethod
    async def build_dataset(cls, name: str, *args, **kwargs) -> np.ndarray:
        """make dataset for training"""
        raise NotImplementedError

    @abstractmethod
    def x_transform(self, *args, **kwargs):
        """
        Transform train data into X, invoked by `load_data`. Usually get overriden by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def y_transform(self, *args, **kwargs):
        """tranform data into target, invoked by `load_data`. Usually get overriden by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray):
        """predict the target for X"""
        X_ = self.x_transform(X)
        if self.model is not None:
            return self.model.predict(X_)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, params: None, scoring: None):
        """fit the model to X and y"""
        raise NotImplementedError
