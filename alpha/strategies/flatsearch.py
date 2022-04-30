import itertools
import os
import pickle

import cfg4py
from pymongo import MongoClient

from alpha.config import get_config_dir
from alpha.core.features import moving_average
from alpha.core.vecstore import VecCollection

cfg = cfg4py.init(get_config_dir())


class FlatSearchStrategy:
    def __init__(self, name: str):
        self.spaces = {}
        self.milvus = Milvus(cfg.milvus.host, cfg.milvus.port)
        self.mongo = MongoClient(cfg.mong.dsn)
        for win in [5, 10, 20, 60]:
            for ft in ["1d", "30m"]:
                name = f"{name}-ma-{win}-{ft}"
                self.spaces[name] = VecCollection(name, "L2", 30, self.milvus)

        self.meta_collection = self.mongo.alpha.meta
        self.name = name

        self.wins = [5, 10, 20, 60, 120, 250]
        self.flen = 50

    def create_spaces(self, drop_if_exists=True):
        for vc in self.spaces.values():
            vc.create_collection(drop_if_exists)

    def reset_space(self):
        for vc in self.spaces.values():
            vc.drop_collection()
            vc.create_collection(True)

    def xtransform(self, bars, target_len=-1):
        if target_len == -1:
            xclose = bars["close"]
        else:
            xclose = bars["close"][:-target_len]

        if len(bars) < self.flen + max(self.wins) - 1:
            return None

        xclose = xclose / xclose[0]
        vec = {}
        for win in self.wins:
            ma = moving_average(xclose, win)[-self.flen :]
            vec[win] = ma.tolist()

        return vec

    def build_sample_space(self, datafile: str, target_len: int, ft: str):
        with open(datafile, "rb") as f:
            ds = pickle.load(f)

        for code, pcr, bars in ds["data"]:
            vec = self.xtransform(bars, target_len=target_len)
            ids = {}
            for k, v in vec.items():
                vc = self.spaces[f"{self.name}-ma-{k}-{ft}"]
                _id = vc.insert([v])
                ids[k] = _id

            self.meta_collection.insert(
                {
                    "code": code,
                    "pcr": pcr,
                    "start": bars["frame"][0],
                    "end": bars["frame"][-target_len - 1],
                    "ids": ids,
                }
            )

    def predict(self, bars):
        vec = self.xtransform(bars)
        results = self.space.search_vec([vec], 999)
        print(results)
