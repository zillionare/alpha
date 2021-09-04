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

from alpha.core.features import fillna, moving_average
from alpha.core.smvecstore import SmallSizeVectorStore

logger = logging.getLogger(__name__)


class MorphFeatures:
    def __init__(
        self, frame_type: FrameType, wins=None, flen=10, threshold=1e-3
    ) -> None:
        self.frame_type = frame_type
        self.wins = wins or [5, 10, 20, 60]
        self.flen = flen
        self.threshold = threshold
        self.store = SmallSizeVectorStore(name=f"morph_{frame_type.value}", columns={})

    def load_store(self, ft: FrameType = None, path: str = None) -> None:
        """load store from disk

        each frame type has its own store

        """
        if path is None:
            path = os.path.expanduser(f"~/alpha/data/morph_{ft.value}.pkl")
        with open(path, "rb") as f:
            self.store = pickle.load(f)

    def save_store(self, path: str = None) -> None:
        if path is None:
            path = os.path.expanduser(f"~/alpha/data/morph_{self.frame_type.value}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.store, f)

    async def add_morph_pattern(
        self, code: str, end: Frame, frame_type: FrameType = FrameType.DAY
    ):
        end = tf.shift(arrow.get(end), 0, frame_type)
        n = max(self.wins) + self.flen - 1
        start = tf.shift(end, -n + 1, frame_type)

        sec = Security(code)
        bars = await sec.load_bars(start, end, frame_type)

        close = bars["close"]
        if np.count_nonzero(np.isfinite(close)) < n * 0.9:
            raise ValueError(f"not enough data for {code}")

        close = fillna(close.copy())
        vec = []
        for win in self.wins:
            ma = moving_average(close, win)[-self.flen :]
            vec.extend(ma / ma[0])

        res = self.store.search_vec(vec, threshold=self.threshold, n=1)
        if res and len(res) > 0:
            logger.info("same morph already exists: %s", res["id_"])
            return res["id_"]
        else:
            return self.store.insert([vec])
