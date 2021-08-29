import logging
import os

import arrow
from alpha.utils.data import make_dataset
from sklearn.metrics import mean_absolute_error

import cfg4py
import numpy as np
from alpha.core.features import (
    fillna,
    moving_average,
    polyfit,
    top_n_argpos,
    transform_to_change_pct,
    transform_y_by_change_pct,
    predict_by_moving_average,
)
from alpha.strategies.base_xgboost_strategy import BaseXGBoostStrategy
from alpha.utils.data import DataBunch
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.security import Security

cfg = cfg4py.init()

logger = logging.getLogger(__name__)


class Seven(BaseXGBoostStrategy):
    def __init__(self, *args, **kwargs):
        home = os.path.expanduser(cfg.alpha.data_home)
        super().__init__("Seven", home)

        self.wins = [5, 10, 20, 30, 60]

        self.n_xbars = max(self.wins) * 2
        self.n_ybars = 5
        self.nbuckets = 10

    def x_transform(self, xbars):
        vec = []
        xclose = fillna(xbars["close"].copy())
        for win in self.wins:
            fit_win = max(7, win)
            ma = moving_average(xclose, win)[-fit_win:]
            ma /= ma[0]
            (a, b, c), pmae = polyfit(ma)
            # c should be always 1
            vec.extend([a, b, pmae])

        return vec

    def y_transform(self, ybars, xbars, err_threshold=0.01):
        xclose = fillna(xbars["close"].copy())
        yclose = ybars["close"]

        yn = len(yclose)

        c0 = xclose[-1]

        if abs(min(yclose) / c0 - 1) < abs(max(yclose) / c0 - 1):
            agg = max
        else:
            agg = min

        # fixme: y should be determined by yclose only, not ypred
        for win in self.wins:
            fit_win = max(14, 2 * win)
            pred_close, pmae = predict_by_moving_average(
                xclose[-fit_win:], win, yn, err_threshold=1
            )

            err = mean_absolute_error(yclose, pred_close) / yclose.mean()
            if err < err_threshold:
                y = agg(pred_close) / c0
                return y, self.y_to_bucket(y)

        return 0, self.nbuckets - 1

    def y_to_bucket(self, y):
        if y == 0:
            return self.nbuckets - 1

        y_ = 100 * (y - 1)
        bins = [-15, -10, -5, 0, 5, 10, 15, 20]
        for i, b in enumerate(bins):
            if y_ < b:
                return i

    async def make_dataset(
        self, total: int, notes: str = None, version=None
    ) -> DataBunch:
        transformers = {
            FrameType.DAY: {
                "func": self.x_transform,
                "bars_len": self.n_ybars + self.n_xbars,
            }
        }
        target_transformer = self.y_transform
        target_win = self.n_ybars

        return await make_dataset(
            transformers, target_transformer, target_win, total, nbuckets=self.nbuckets
        )
