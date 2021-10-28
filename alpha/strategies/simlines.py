import datetime
from arrow import Arrow
import arrow
import numpy as np
from omicron.core.timeframe import tf
from typing import List, NewType
from alpha.core.features import fillna, moving_average
from omicron.models.securities import Securities
from omicron.models.security import Security
from sklearn.metrics.pairwise import paired_euclidean_distances
import pandas as pd

from omicron.core.types import FrameType, SecurityType

Frame = NewType("Frame", (str, datetime.date, datetime.datetime, Arrow))


class SimLines:
    def __init__(
        self, sample_code: str, sample_end: Frame, feat_len: int, ma_groups: List
    ):
        self.sample_code = sample_code
        self.sample_end = sample_end
        self.feat_len = feat_len
        self.ma_groups = ma_groups
        self.frame_type = FrameType.DAY

        self.features = []
        self._results = None

    async def init(self):
        n = max(self.ma_groups) + self.feat_len - 1
        end = arrow.get(self.sample_end)
        start = tf.day_shift(end, -n + 1)

        bars = await Security(self.sample_code).load_bars(start, end, self.frame_type)
        self.features = self.extract_features(bars)

    def extract_features(self, bars: np.ndarray):
        features = []

        close = bars["close"]
        if np.count_nonzero(np.isfinite(close)) < len(bars) * 0.9:
            raise Exception("Close contains too many nan.")

        close = fillna(close.copy())
        for win in self.ma_groups:
            ma = moving_average(close, win)[-self.feat_len :]
            ma /= ma[0]

            features.append(ma.tolist())

        return features

    async def find_similar(self, end: Frame):
        """find security which ma lines are similar to the sample at `end`

        Args:
            end (Frame): [description]
        """
        n = max(self.ma_groups) + self.feat_len - 1

        self._results = []
        end = arrow.get(end)
        start = tf.shift(end, -n + 1, self.frame_type)
        for code in Securities().choose(["stock"]):
            try:
                sec = Security(code)
                bars = await sec.load_bars(start, end, self.frame_type)
                features = self.extract_features(bars)

                d = paired_euclidean_distances(features, self.features)
                self._results.append([sec.display_name, code, d, sum(d), np.var(d)])
            except Exception:
                pass

        return pd.DataFrame(
            self._results, columns=["name", "code", "distance", "sum", "var"]
        ).sort_values("sum")

    async def backtest(self, start: Frame, end: Frame):
        """从start`到`end`之间进行回测

        Args:
            start (Frame): [description]
            end (Frame): [description]
        """
        start = arrow.get(start)
        end = arrow.get(end)

        if self.frame_type in tf.day_level_frames:
            convertor = tf.int2date
        else:
            convertor = tf.int2datetime

        orders = []
        for dt in tf.get_frames(start, end, self.frame_type):
            tail = convertor(dt)
            await self.find_similar(tail)
            orders.extend(self._results)

        # find best results
        for long_order in orders:
            name, code, frame, buy_price = long_order
            pass
