import logging
import os

import arrow
import cfg4py
import numpy as np
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.security import Security

from alpha.core.features import (
    fillna,
    ma_permutation,
    top_n_argpos,
    transform_to_change_pct,
    transform_y_by_change_pct,
)
from alpha.strategies.base_xgboost_strategy import BaseXGBoostStrategy
from alpha.strategies.databunch import DataBunch

cfg = cfg4py.init()

logger = logging.getLogger(__name__)


class Z09(BaseXGBoostStrategy):
    """same as z08, but target will have only two value: -1 and 1

        if tomorrow's close is greate than today's close, the target is 1, otherwise, -1

        final result
                  precision    recall  f1-score   support

              -1       0.55      0.58      0.57       500
               1       0.56      0.53      0.54       500

        accuracy                           0.56      1000
       macro avg       0.56      0.55      0.55      1000
    weighted avg       0.56      0.56      0.55      1000

    """

    def __init__(self):
        home = os.path.expanduser(cfg.alpha.data_home)
        super().__init__("Z09", home, "classifier", eval_metric="binary:logistic")

        self.n_features = 10
        self.target_win = 1
        self.max_ma_win = 250

    async def make_dataset(self, total: int, notes: str = None) -> DataBunch:
        now = arrow.now()

        bars_len = self.max_ma_win + self.n_features + self.target_win
        count_of_secs = 3000
        samples_per_sec = int(total / count_of_secs) + 1

        end = tf.day_shift(now, -10)
        start = tf.day_shift(end, -samples_per_sec * 5)

        X = []
        y = []
        label_counters = {1: 0, -1: 0}

        for code, tail in self.dataset_scope(total * 5, start, end, FrameType.DAY):
            sec = Security(code)

            head = tf.day_shift(tail, -bars_len + 1)
            bars = await sec.load_bars(head, tail, FrameType.DAY)

            if len(bars) != bars_len:
                continue

            ybars = bars[-self.target_win :]
            xbars = bars[: -self.target_win]

            if ybars["close"][0] and xbars["close"][-1]:
                y_ = 1 if ybars["close"][0] > xbars["close"][-1] else -1
                if label_counters[y_] > total / 2 + 1:
                    continue
            else:  # there's no y_ in this case
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
            mclose = min_bars["close"]
            for vol_win in (5, 10, 20):
                volume = min_bars["volume"][-vol_win:]
                top_1, top_2 = top_n_argpos(volume, 2)

                vol_top_1, vol_top_2 = volume[top_1], volume[top_2]

                # pos from very first of whole min_bars, thus we can calc top_*_flag by price
                _top_1 = len(min_bars) - vol_win + top_1
                _top_2 = len(min_bars) - vol_win + top_2

                top_1_flag = 1 if mclose[_top_1] / mclose[_top_1 - 1] >= 1 else -1
                top_2_flag = 1 if mclose[_top_2] / mclose[_top_2 - 1] >= 1 else -1

                if top_1_flag == top_2_flag:
                    row.append(top_1_flag)
                elif vol_top_1 >= 2 * vol_top_2:
                    row.append(top_1_flag)
                else:
                    row.append(0)

            X.append(row)
            y.append(y_)

            label_counters[y_] += 1
            if len(X) % 100 == 0:
                logger.info(
                    "%s/%s samples collected: %s", len(X), total, label_counters
                )

            if len(X) >= total:
                break

        desc = notes
        ds = DataBunch(name=self.name.lower(), X=X, y=y, desc=desc, raw=bars)

        logger.info("%s created with %s", ds.name, ds.X.shape)
        return ds