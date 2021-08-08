from cProfile import label
import os
from alpha.strategies.databunch import DataBunch
import pickle
from configparser import NoOptionError
from ctypes import Union
from typing import Callable

import arrow
import numpy as np
from numpy.typing import ArrayLike
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security

from alpha.core.features import (
    fillna,
    ma_permutation,
    moving_average,
    pos_encode,
    transform_y_by_change_pct,
)
from alpha.strategies.base_xgboost_strategy import BaseXGBoostStrategy
import cfg4py

cfg = cfg4py.init()


class Z04(BaseXGBoostStrategy):
    """使用以下特征：
    1. 均线排列情况。使用5, 10, 20, 30, 60, 120, 250最近7天的排列情况（使用pos_encode编码）
    2. 250日内、120日内、60日内最高量、最低量发生日期，及距今日的天数，除以250,120,60以归一化
    3. 最近10天的价格变化率（0.01~0.2），不做归一化
    4. 30分钟线的均线排列情况，使用5, 10, 20, 30(pos_encode编码）

    target: 如果未来5天出现收盘价大于当日5%，则认为上涨（1）；如果小于当日-5%，则认为下跌（-1），否则为0。


    Args:
        BaseXGBoostStrategy ([type]): [description]
    """

    def __init__(self):
        name = "Z04"
        base_model = "classifier"
        home = os.path.expanduser(cfg.alpha.data_home)
        super().__init__(name, home, base_model)

    async def make_dataset(self, code: str, total: int):
        """

        Args:
            save_to (str): [description]
        """
        now = arrow.now()
        end = tf.day_shift(now, -30)
        n_features = 10

        start = tf.day_shift(end, -total - n_features - 260)

        sec = Security(code)
        bars = await sec.load_bars(start, end, FrameType.DAY)
        if len(bars) < total:
            raise Exception("not enough data")

        X = []
        y = []

        label_counters = {1: 0, 0: 0, -1: 0}

        close = fillna(bars["close"].copy())

        for i in range(total):
            train = close[i : i + 250 + n_features]
            t_start = i + 250 + n_features
            target = close[t_start - 1 : t_start + 3]

            y_ = transform_y_by_change_pct(target, (0.95, 1.05))
            if y_ is None:
                continue

            # keep all labels balanced
            label_counters[y_] += 1
            if label_counters[y_] >= total / 3 + 1:
                continue

            x = ma_permutation(train, n_features, [5, 10, 20, 30, 60, 120, 250])

            X.append(x)
            y.append(y_)

        desc = (
            f"dataset for strategy Z04. The dataset includes bars from {start}"
            f"to {end}, {len(bars)} in total. \n"
            f"params: samples {len(X)}, features {n_features}, ma of [5, 10, 20, 30, 60, 120, 250]. Feature labels are balanced."
        )

        ds = DataBunch(name="z04", X=np.array(X), y=np.array(y), raw=bars, desc=desc)

        return ds
