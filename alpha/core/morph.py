"""given code, start, frame_type and nframes, capture its morphal features, stored as features.

To simpify the morphal patterns, add these limitations:

1. use ma of (20, 30, 60) only
2. nframes of each feature is 20

from high score to low score, we'll have 5 kinds:

2: 已经出现大涨后的均线、price线
1：均线多头向上，还未进入最后大涨
0：整理阶段
-1：
-2:

"""

import datetime
import os
import pickle
from typing import List, NewType, Union

import arrow
from omicron.models.security import Security

# define new types
Frame = NewType("Frame", (str, datetime.date, datetime.datetime))

import logging

import numpy as np
from omicron.core.timeframe import tf
from omicron.core.types import FrameType

from alpha.core.features import fillna, moving_average, weighted_moving_average
from alpha.core.smvecstore import SmallSizeVectorStore

logger = logging.getLogger(__name__)


class MorphFeatures:
    def __init__(
        self, frame_type: FrameType, wins=None, flen=10, thresholds=None
    ) -> None:
        self.frame_type = frame_type
        self.wins = wins or [5, 10, 20, 60]
        self.flen = flen
        self.thresholds = thresholds or {
            5: 1e-2,
            10: 7e-3,
            20: 5e-3,
            60: 3e-3,
        }

        self.default_threshold = 1e-3

        self.version = 1

        self.stores = {}
        for win in self.wins:
            self.stores[win] = SmallSizeVectorStore(f"morph_{frame_type.value}_{win}")

    def __str__(self) -> str:
        desc = f"FrameType: {self.frame_type.value}\n"
        desc += f"thresholds:\n{self.thresholds}\n"
        for win in self.wins:
            desc += f"{win}: {len(self.stores[win])}\n"

        return desc

    def __repr__(self) -> str:
        return super().__repr__() + f" Ver: v{self.version}"

    def get_threshold(self, win: int) -> float:
        if win in self.thresholds:
            return self.thresholds[win]
        else:
            return self.default_threshold

    @staticmethod
    def load(ft: FrameType = None, path: str = None) -> None:
        """load store from disk

        each frame type has its own store

        """
        if path is None:
            path = os.path.expanduser(f"~/alpha/data/morph_{ft.value}.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

    def dump(self, path: str = None) -> None:
        """save store to disk

        before saving, each store will be sorted,thus their id_ will be continuous related to vector features.

        as a result, the store could be different between two dumps, especialy regarding to ids.
        """
        self.version += 1

        for win in self.wins:
            # sort the vecs by last dim, since they are already normailized to start with 1
            self.stores[win] = self.stores[win].sorted(lambda x: np.argsort(x[:, -1]))

        if path is None:
            path = os.path.expanduser(f"~/alpha/data/morph_{self.frame_type.value}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    async def encode(
        self, code: str, end: Frame, frame_type: FrameType = FrameType.DAY
    ):
        """transform moving average trend line into morph features

        Args:
            code (str): [description]
            end (Frame): [description]
            frame_type (FrameType, optional): [description]. Defaults to FrameType.DAY.

        Raises:
            ValueError: [description]
        """
        end = tf.shift(arrow.get(end), 0, frame_type)
        n = max(self.wins) + self.flen - 1
        start = tf.shift(end, -n + 1, frame_type)

        sec = Security(code)
        bars = await sec.load_bars(start, end, frame_type)

        close = bars["close"]
        if np.count_nonzero(np.isfinite(close)) < n * 0.9 or not np.all(
            np.isfinite(close[-3:])
        ):
            raise ValueError(f"not enough data for {code}")

        close = fillna(close.copy())

        features = []
        for win in self.wins:
            ma = weighted_moving_average(close, win)[-self.flen :]
            vec = ma / ma[0]

            store = self.stores[win]
            res = store.nearest_vec(vec, threshold=self.get_threshold(win), n=1)
            if res is not None and len(res) > 0:
                features.append(res["id_"][0])
            else:
                features.append(store.insert([vec])[0])

        return features
