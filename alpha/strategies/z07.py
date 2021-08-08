from alpha.core.features import (
    transform_to_change_pct,
    fillna,
    ma_permutation,
    transform_y_by_change_pct,
)
from alpha.strategies.base_xgboost_strategy import BaseXGBoostStrategy
from alpha.strategies.databunch import DataBunch
import os
import cfg4py
import arrow
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.security import Security
import numpy as np
import logging

cfg = cfg4py.init()

logger = logging.getLogger(__name__)


class Z07(BaseXGBoostStrategy):
    """all features from Z06, plus adv/dec flag indicated by max volume in the last 5, 10, 20 frames.

    Args:
        Z05 ([type]): [description]
    """

    def __init__(self):
        home = os.path.expanduser(cfg.alpha.data_home)
        super().__init__("Z07", home, "classifier")

        self.n_features = 10
        self.target_win = 3
        self.max_ma_win = 250

    async def make_dataset(self, total: int, notes: str=None) -> DataBunch:
        now = arrow.now()

        bars_len = self.max_ma_win + self.n_features + self.target_win
        count_of_secs = 3000
        samples_per_sec = int(total / count_of_secs) + 1

        end = tf.day_shift(now, -10)
        start = tf.day_shift(end, -samples_per_sec * 5)

        X = []
        y = []
        label_counters = {1: 0, 0: 0, -1: 0}

        for code, tail in self.dataset_scope(total * 5, start, end, FrameType.DAY):
            sec = Security(code)

            head = tf.day_shift(tail, -bars_len + 1)
            bars = await sec.load_bars(head, tail, FrameType.DAY)

            if len(bars) != bars_len:
                continue

            ybars = bars[-self.target_win :]
            xbars = bars[: -self.target_win]

            close = bars["close"]

            # to calc the advance/decrease rate, we need c0 and all close in target_win
            y_ = transform_y_by_change_pct(
                ybars["close"], (0.95, 1.05), xbars[-1]["close"]
            )
            if y_ is None or label_counters[y_] > total / 3 + 1:
                continue

            try:
                xclose = fillna(xbars["close"].copy())
            except ValueError:
                continue

            assert len(xclose) == self.max_ma_win + self.n_features

            min_end_day = xbars["frame"][-1]
            min_frame_end = tf.combine_time(min_end_day, 15)
            # use [5, 10, 20] for min frames
            min_frame_start = tf.shift(min_frame_end, -30, FrameType.MIN30)

            min_bars = await sec.load_bars(
                min_frame_start, min_frame_end, FrameType.MIN30
            )
            min_close = min_bars["close"]
            if np.count_nonzero(np.isnan(min_close)) > 0:
                logger.debug("no close data in %s~%s", min_frame_start, min_frame_end)
                continue

            samples_made = sum(label_counters.values())
            if samples_made % 100 == 0:
                logger.info(
                    "%s/%s samples collected: %s", samples_made, total, label_counters
                )

            row = ma_permutation(xclose, 10, [5, 10, 20, 30, 60, 120, 250])
            row.extend(ma_permutation(min_close, 10, [5, 10, 20]))
            row.extend(transform_to_change_pct(xclose[-6:]))
            row.extend(transform_to_change_pct(min_close[-21:]))

            # directrion by max(vol) in 5, 10, 20 frames
            for vol_win in (5, 10, 20):
                volume = xbars["volume"][-vol_win:]
                pos = len(xbars) - vol_win + np.argmax(volume)
                direction = 1 if xclose[pos] / xclose[pos - 1] >= 1 else -1
                row.append(direction)

            X.append(row)
            y.append(y_)

            label_counters[y_] +=1
            if len(X) % 100 == 0:
                logger.info("%s/%s samples collected: %s", len(X), total, label_counters)

            if len(X) >= total:
                break

        desc = notes
        ds = DataBunch(name=self.name.lower(), X=X, y=y, desc=desc, raw=bars)

        logger.info("%s created with %s", ds.name, ds.X.shape)
        return ds
