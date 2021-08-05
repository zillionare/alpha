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


class Z04(BaseXGBoostStrategy):
    """使用以下特征：
    1. 均线排列情况。使用5, 10, 20, 30, 60, 120, 250最近7天的排列情况（使用pos_encode编码）
    2. 250日内、120日内、60日内最高量、最低量发生日期，及距今日的天数，除以250,120,60以归一化
    3. 最近10天的价格变化率（0.01~0.2），不做归一化
    4. 30分钟线的均线排列情况，使用5, 10, 20, 30(pos_encode编码）

    target: 如果未来3天出现收盘价大于当日5%，则认为上涨（1）；如果小于当日-5%，则认为下跌（-1），否则为0。


    Args:
        BaseXGBoostStrategy ([type]): [description]
    """

    def __init__(self):
        name = "Z04"
        base_model = "classifier"
        super().__init__(name, base_model=base_model)

    async def build_dataset(self, code: str, total: int, save_to: str):
        """

        Args:
            save_to (str): [description]
        """
        now = arrow.now()
        end = tf.day_shift(now, -30)
        start = tf.day_shift(end, -total - 260)

        sec = Security(code)
        bars = await sec.load_bars(start, end, FrameType.DAY)
        if len(bars) < total:
            raise Exception("not enough data")

        X = []
        y = []
        # 均线排列情况 win = 7
        win = 7
        close = fillna(bars["close"].copy())

        for i in range(total):
            train = close[i : i + 250 + win]
            t_start = i + 250 + win
            target = close[t_start - 1 : t_start + 3]

            x = self.x_transform(train)
            y_ = self.y_transform(target)

            if y_ is not None:
                X.append(x)
                y.append(y_)

        ds = {"X": np.array(X), "y": np.array(y), "raw": bars, "code": code}

        with open(save_to, "wb") as f:
            pickle.dump(ds, f)

    def x_transform(self, close: ArrayLike):
        """
        Args:
            bars (list): [description]
        """
        assert len(close) == 257

        ma5 = moving_average(close, 5)[-7:]
        ma10 = moving_average(close, 10)[-7:]
        ma20 = moving_average(close, 20)[-7:]
        ma30 = moving_average(close, 30)[-7:]
        ma60 = moving_average(close, 60)[-7:]
        ma120 = moving_average(close, 120)[-7:]
        ma250 = moving_average(close, 250)[-7:]

        stationary = np.array([5, 10, 20, 30, 60, 120, 250])

        codes = []
        for i in range(7):
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

    def get_X(self, dataset: str):
        if dataset == "train":
            return self.data_train[0]
        elif dataset == "valid":
            return self.data_valid[0]
        else:
            return self.data_test[0]

    def get_y(self, dataset: str):
        if dataset == "train":
            return self.data_train[1]
        elif dataset == "valid":
            return self.data_valid[1]
        else:
            return self.data_test[1]
