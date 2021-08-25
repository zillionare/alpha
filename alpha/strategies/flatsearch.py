import os
import pickle

import cfg4py
from alpha.config import get_config_dir
from alpha.core.features import moving_average
from alpha.core.vecstore import VecStore
from pymilvus import Milvus

cfg = cfg4py.init(get_config_dir())


class FlatSearchStrategy:
    def __init__(self, name: str, create_space: bool = True):
        self.space = VecStore(
            name,
            "L2",
            28,
            milhost=cfg.milvus.host,
            milport=cfg.milvus.port,
            meta_dsn=cfg.mongo.dsn,
        )
        self.wins = [5, 10, 20, 60]
        self.flen = 7
        self.space.create_collection(False)

    def reset_space(self):
        self.space.drop_collection()
        self.space.create_collection(False)

    def xtransform(self, bars, target_len=-1):
        if target_len == -1:
            xclose = bars["close"]
        else:
            xclose = bars["close"][:-target_len]

        xclose = xclose[-max(self.wins) - self.flen :]
        xclose = xclose / xclose[-1]
        vec = []
        for win in [5, 10, 20, 60]:
            ma = moving_average(xclose, win)[-self.flen :]
            vec.extend(ma.tolist())

        return vec

    def build_sample_space(self, datafile: str):
        with open(datafile, "rb") as f:
            ds = pickle.load(f)

        for code, pcr, bars in ds["data"][:100]:
            self.space.insert(
                [self.xtransform(bars, target_len=10)],
                meta=[
                    {
                        "code": code,
                        "pcr": pcr,
                        "start": bars["frame"][0],
                        "end": bars["frame"][-1],
                    }
                ],
            )

    def predict(self, bars):
        vec = self.xtransform(bars)
        results = self.space.search_vec([vec], 999)
        print(results)
