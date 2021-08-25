"""given code, start, frame_type and nframes, capture its morphal features, stored as features.

To simpify the morphal patterns, add these limitations:

1. use ma of (5, 10 ,20) only
2. nframes of each feature is 20

from high score to low score, we'll have 5 kinds:

2: 已经出现大涨后的均线、price线
1：均线多头向上，还未进入最后大涨
0：整理阶段
-1：
-2:

"""

import os
import pickle
from sqlite3 import DatabaseError, DataError
from typing import List

import numpy as np
from alpha.core.features import moving_average
from alpha.core.smvecstore import SmallSizeVectorStore


class MorphaFeatures:
    def __init__(self) -> None:
        self.wins = [5, 10, 20]
        self.samples = 5
        self.flen = 7
        self.stores = {}

        self.nbars = max(self.wins) + self.flen + self.samples - 1
        for win in self.wins:
            name = f"store_ma{win}"
            params = {"acc": "<f4", "b": "<f4", "pcr": "<f4", "vx": "<f4"}
            store = SmallSizeVectorStore(name, params)

            self.stores[win] = store

        self.build()

    def scale(self, x):
        if min(x) < 1:
            x += -min(x) + 1

        return x

    def build(self) -> None:
        """build features for model"""
        for a in [i / 20000 for i in range(-20, 20)]:
            # b= 0.1对应每天涨停的情况
            for b in [i / 200 for i in range(-20, 20)]:
                p = np.poly1d((a, b, 1))

                # use same price serices for all ma2
                y = p(np.arange(self.nbars))
                y = self.scale(y)

                # vx is the distance to the end of bars
                vx = self.nbars - 1 - (-b) / (2 * (a or 1e-13))
                vx = np.tanh(vx * 2 / self.nbars)

                for win in self.wins:
                    store = self.stores[win]
                    ma = moving_average(y, win)
                    for j in range(-self.samples + 1, 1):
                        vec = ma[len(ma) + j - self.flen : len(ma) + j]
                        pcr = ma[len(ma) - 1 + j] / ma[len(ma) + j - 2] - 1
                        store._insert_one(
                            {
                                "acc": a,
                                "b": b,
                                "pcr": pcr,
                                "vx": vx,
                            },
                            vec,
                        )

    def xtransform(self, close: np.ndarray) -> List:
        """transform x to features

        :param x:
        :return:
        """
        vec = []

        if len(close) < self.nbars:
            raise DataError("not enough data")

        close = close[-self.nbars :]

        if np.count_nonzero(np.isfinite(close)) != len(close):
            raise DataError("data contains np.NaN or None")

        close = self.scale(close / close[0])
        dtypes = None
        for win in self.wins:
            ma = moving_average(close, win)[-self.flen :]
            matched = self.stores[win].search_vec(ma, threshold=999, n=1)
            dtypes = matched.dtype
            if matched and len(matched) == 1:
                # acc, b, pcr, vx, d(distance)
                vec.append(matched[0].tolist())

        return np.array(vec, dtype=dtypes)
