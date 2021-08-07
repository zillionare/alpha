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

from alpha.core.features import fillna, moving_average, pos_encode
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

        start = tf.day_shift(end, -total -n_features - 260)

        sec = Security(code)
        bars = await sec.load_bars(start, end, FrameType.DAY)
        if len(bars) < total:
            raise Exception("not enough data")

        X = []
        y = []

        label_counters = {
            1: 0,
            0: 0,
            -1: 0
        }

        close = fillna(bars["close"].copy())

        for i in range(total):
            train = close[i : i + 250 + n_features]
            t_start = i + 250 + n_features
            target = close[t_start - 1 : t_start + 3]

            y_ = self.y_transform(target)
            if y_ is None:
                continue

            # keep all labels balanced
            label_counters[y_] += 1
            if label_counters[y_] >= total / 3 + 1:
                continue

            x = self.x_transform(train, n_features)

            X.append(x)
            y.append(y_)

        desc = f"dataset for strategy Z04. The dataset includes bars from {start}" \
        f"to {end}, {len(bars)} in total. \n" \
        f"params: samples {len(X)}, features {n_features}, ma of [5, 10, 20, 30, 60, 120, 250]. Feature labels are balanced."

        ds = DataBunch(name="z04", X = np.array(X), y = np.array(y), raw = bars, desc=desc)

        return ds

    def x_transform(self, close: ArrayLike, n_features: int):
        """
        Args:
            bars (list): [description]
        """
        ma5 = moving_average(close, 5)[-n_features:]
        ma10 = moving_average(close, 10)[-n_features:]
        ma20 = moving_average(close, 20)[-n_features:]
        ma30 = moving_average(close, 30)[-n_features:]
        ma60 = moving_average(close, 60)[-n_features:]
        ma120 = moving_average(close, 120)[-n_features:]
        ma250 = moving_average(close, 250)[-n_features:]

        stationary = np.array([5, 10, 20, 30, 60, 120, 250])

        codes = []
        for i in range(n_features):
            ma_list = np.array(
                [ma5[i], ma10[i], ma20[i], ma30[i], ma60[i], ma120[i], ma250[i]]
            )
            pos = np.argsort(ma_list)

            codes.append(pos_encode(stationary, stationary[pos]))

        return codes

    def y_transform(self, close):
        """取三日内最大涨跌幅（收盘价计）

        Args:
            bars ([type]): [description]
        """
        assert len(close) == 4

        c0 = close[0]
        if c0 == np.NaN or np.all(close[1:] == np.NaN):
            return NoOptionError

        # 止损优先
        if min(close[1:]) / c0 <= 0.95:
            return -1
        elif max(close[1:]) / c0 >= 1.05:
            return 1
        else:
            return 0
