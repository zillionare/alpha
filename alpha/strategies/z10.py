from xml.sax.handler import feature_string_interning
from omicron.models.securities import Securities

from sklearn.metrics import make_scorer, max_error, mean_absolute_error
from alpha.core.features import fillna, moving_average, relative_strength_index
from typing import List
from alpha.core.errors import NoFeaturesError, NoTargetError
from alpha.strategies.databunch import DataBunch
import os
import cfg4py
import arrow
import numpy as np
import logging
import datetime

from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.security import Security
from alpha.strategies.base_xgboost_strategy import BaseXGBoostStrategy

from alpha import utils

logger = logging.getLogger(__name__)

cfg = cfg4py.get_instance()


class Z10(BaseXGBoostStrategy):
    def __init__(self):
        home = os.path.expanduser(cfg.alpha.data_home)
        super().__init__("Z10", home)
        self.target_win = 1

    async def make_dataset(
        self, total: int, desc: str = None, version: str = None
    ) -> DataBunch:
        target_win = 1
        transformers = {
            FrameType.DAY: {"bars_len": 260, "func": self.x_transform},
            FrameType.MIN30: {"bars_len": 40, "func": self.x_transform},
        }
        target_transformer = self.y_transform

        bucket_size = 21
        capacity = total // 50

        ds = await utils.data.make_dataset(
            transformers,
            target_transformer,
            target_win,
            (bucket_size, capacity, self.is_sample_enough),
            secs = self.choose_secs()
        )

        ds.desc = desc
        ds.name = self.__class__.__name__
        ds.version = version

        return ds

    def choose_secs(self):
        secs = Securities()

        result = []

        for code in secs.choose(['stock']):
            if code.startswith('688'):
                continue
            if code.startswith('3'):
                continue

            result.append(code)

        return result

    def y_transform(self, ybars: np.array, xbars: np.array) -> float:
        c1 = ybars[0]["close"]
        c0 = xbars[-1]["close"]

        if np.all(np.isfinite([c1, c0])) and (c0 != 0):
            return c1 / c0 - 1
        else:
            raise NoTargetError

    def x_transform(self, bars: np.array, flen: int = 7) -> List:
        x = np.arange(flen)

        close = bars["close"].copy()
        if np.count_nonzero(np.isfinite(close)) < len(close) * 0.9:
            raise NoFeaturesError

        close = fillna(close)

        results = []
        # add rsi
        rsi = relative_strength_index(close, 5)[-flen:]
        results.extend(rsi/100)

        for col in ("open", "close", "high", "low"):
            price = bars[col]
            if np.count_nonzero(np.isfinite(price)) < len(price) * 0.9:
                raise NoFeaturesError

            price = fillna(price.copy())
            price = (price[1:] / price[:-1] - 1)[-flen:]

            (a, b, _), (err, *_), *_ = np.polyfit(x, price, 2, full=True)

            results.extend((a, b, err))

        for win in (5, 10, 20, 30):
            ma = moving_average(close, win)
            ma = (ma[1:] / ma[:-1] - 1)[-flen:]

            (a, b, _), (err, *_), *_ = np.polyfit(x, ma, 2, full=True)

            results.extend((a, b, err))

        return results

    def y_limit(self, code, frame):
        if code.startswith("688") or (
            code.startswith("3") and frame > datetime.date(2020, 1, 1)
        ):
            return 1
        else:
            return 0

    def is_sample_enough(self, buckets, y, capacity):
        idx = int(y * 100) + 10
        if buckets[idx] < capacity:
            buckets[idx] += 1
            return False
        else:
            return True