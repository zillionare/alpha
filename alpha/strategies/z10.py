from xml.sax.handler import feature_string_interning

from sklearn.metrics import make_scorer, max_error, mean_absolute_error
from alpha.core.features import fillna, moving_average
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


logger = logging.getLogger(__name__)

cfg = cfg4py.get_instance()


class Z10(BaseXGBoostStrategy):
    def __init__(self):
        home = os.path.expanduser(cfg.alpha.data_home)
        super().__init__("Z10", home)
        self.target_win = 1

    async def make_dataset(self, total: int, note: str = None) -> DataBunch:

        day_bars_len = 50
        count_of_secs = 3000
        samples_per_sec = int(total / count_of_secs) + 1

        end = tf.day_shift(arrow.now(), -10)
        start = tf.day_shift(end, -samples_per_sec * 10)

        X, y = [], []
        y_distribution = [0] * 42
        for code, tail in self.dataset_scope(start, end):
            sec = Security(code)

            head = tf.day_shift(tail, -day_bars_len + 1)
            bars = await sec.load_bars(head, tail, FrameType.DAY)

            if len(bars) != day_bars_len:
                continue

            ybars = bars[-self.target_win :]
            xbars = bars[: -self.target_win]

            try:
                y_ = self.y_transform(ybars["close"][0], xbars["close"][-1])
            except NoTargetError:
                continue

            if self.has_enough_samples(y_distribution, y_, total / 40):
                continue

            try:
                day_features = self.x_transform(xbars)
            except NoFeaturesError:
                continue

            try:
                # handle MIN30 level bars
                tail = xbars[-1]["frame"]
                tail = tf.combine_time(tail, hour=15)

                head = tf.shift(tail, -50, FrameType.MIN30)
                mbars = await sec.load_bars(head, tail, FrameType.MIN30)
                m_features = self.x_transform(mbars)
            except NoFeaturesError:
                continue

            x = day_features
            x.extend(m_features)
            x.append(self.y_limit(code, xbars[-1]["frame"]))
            X.append(x)
            y.append(y_)

            if len(X) >= total:
                break

            if len(X) % 100 == 0:
                logger.info(
                    "%s/%s samples collected: %s", len(X), total, y_distribution
                )

        ds = DataBunch(name=self.name.lower(), X=X, y=y, desc=note)
        logger.info("%s created with %s", ds.name, ds.X.shape)
        return ds

    def y_transform(self, c1: float, c0: float):
        if np.all(np.isfinite([c1, c0])) and (c0 != 0):
            return c1 / c0 - 1
        else:
            raise NoTargetError

    def x_transform(self, bars: np.array, flen: int = 7) -> List:
        x = np.arange(flen)

        results = []
        for col in ("open", "close", "high", "low"):
            price = bars[col]
            if np.count_nonzero(np.isfinite(price)) < len(price) * 0.9:
                raise NoFeaturesError

            price = fillna(price.copy())
            price = (price[1:] / price[:-1] - 1)[-flen:]

            (a, b, _), (err, *_), *_ = np.polyfit(x, price, 2, full=True)

            results.extend((a, b, err))

        close = bars["close"].copy()
        close = fillna(close)

        for win in (5, 10, 20, 30):
            ma = moving_average(close, win)
            ma = (ma[1:] / ma[:-1] - 1)[-flen:]

            (a, b, _), (err, *_), *_ = np.polyfit(x, ma, 2, full=True)

            results.extend((a, b, err))

        return results

    def has_enough_samples(self, y_distribution, y_, count):
        idx = int(y_ * 100) + 20

        if y_distribution[idx] >= count:
            return True
        else:
            y_distribution[idx] += 1
            return False


    def y_limit(self, code, frame):
        if code.startswith("688") or (
            code.startswith("3") and frame > datetime.date(2020, 7, 20)
        ):
            return 1
        else:
            return 0

    def fit(self, ds: DataBunch, params=None):
        scoring = make_scorer(max_error, greater_is_better=False)
        super().fit(ds, params, scoring=scoring)
