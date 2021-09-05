import logging
import os
from random import sample

import arrow
import cfg4py
import numpy as np
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.security import Security

from alpha.core.features import fillna, ma_permutation, transform_y_by_change_pct
from alpha.strategies.base_xgboost_strategy import BaseXGBoostStrategy
from alpha.utils.data import DataBunch

logger = logging.getLogger(__name__)

cfg = cfg4py.init()


class Z05(BaseXGBoostStrategy):
    """
    Strategy Z05 use both day-level and 30-min level ma parrallel as features
    """

    def __init__(self):
        """
        Initializing the strategy
        """
        home = os.path.expanduser(cfg.alpha.data_home)

        super().__init__("Z05", home, "classifier")

    async def make_dataset(self, total: int, notes: str = "") -> DataBunch:
        now = arrow.now()

        n_features = 10
        target_win = 3
        max_ma_win = 250

        bars_len = max_ma_win + n_features + target_win

        # assuming A-share market has 3000 securities fit for this dataset.
        count_of_secs = 3000
        # how many bars should we sample from each security?
        n_bars = int(total / count_of_secs) + 1

        end = tf.day_shift(now, -10)
        start = tf.day_shift(end, -n_bars * 5)

        X = []
        y = []
        label_counters = {1: 0, 0: 0, -1: 0}

        # multi total by 5, to leave room for: balancing labels, stop listing days, ...
        for code, tail in self.dataset_scope(total * 5, start, end, FrameType.DAY):
            sec = Security(code)

            head = tf.day_shift(tail, -bars_len + 1)
            bars = await sec.load_bars(head, tail, FrameType.DAY)

            if len(bars) != bars_len:
                continue

            close = bars["close"]

            # to calc the advance/decrease rate, we need c0 and all close in target_win
            target = close[-(target_win + 1) :]
            y_ = transform_y_by_change_pct(target, (0.95, 1.05))
            if y_ is None or label_counters[y_] > total / 3 + 1:
                continue

            label_counters[y_] += 1

            try:
                close = fillna(close.copy())
            except ValueError:
                continue

            train = close[0:-target_win]
            assert len(train) == max_ma_win + n_features

            min_end_day = bars["frame"][-(target_win + 1)]
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

            row = ma_permutation(train, 10, [5, 10, 20, 30, 60, 120, 250])
            row.extend(ma_permutation(min_close, 10, [5, 10, 20]))

            X.append(row)
            y.append(y_)

            if len(X) >= total:
                break

        X = np.array(X)
        y = np.array(y)

        assert len(X) == len(y)

        desc = (
            f"Dataset for strategy Z05. The dataset includes bars from {start}"
            f"to {end}, {len(bars)} in total.\n"
            f"shape: {X.shape}, use ma of [5, 10, 20, 30, 60, 120, 250] for day bars and [5, 10, 20] for 30mins bars. Feature labels are balanced on day bars"
        )

        ds = DataBunch(
            name="z05",
            X=X,
            y=y,
            desc=desc + f"\n{notes}",
            raw={"day": bars, "min30": min_bars},
        )

        logger.info("%s created with %s", ds.name, ds.X.shape)
        return ds
