import json

import pandas as pd
from alpha.features.volume import top_volume_direction
from alpha.core.features import fillna, moving_average, polyfit, predict_by_moving_average, relation_with_prev_high, relative_strength_index
import datetime
from doctest import FAIL_FAST
from typing import List, NewType
import arrow
from arrow import Arrow
import numpy as np
import omicron
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security
import logging


Frame = NewType("Frame", (datetime.date, datetime.datetime, str, Arrow))

logger = logging.getLogger(__name__)

class UpTurn:
    """
    寻找5周期均线向上拐头（加速度a > 0),预期5周期收益 > profit，并且其它周期也在拐头的股票。

    找到的股票类似以下形态：
    1. 下一周期上涨。最短周期均线已经拐头
    2. 其它周期也在拐头或者止跌中 （a >= 0)，并且如果上涨趋势持续，这些周期的均线也最终将向上
    """
    def __init__(self)->None:
        self.name = self.__class__.__name__.lower()
        self.ma_min_wins = [5, 10, 20, 60]
        self.n_minbars = max(self.ma_min_wins) + 20

        self.ma_day_wins = [5, 10, 20, 60]
        self.n_daybars = max(self.ma_day_wins) + 20

        self.X = []

    def get_pmae_err_threshold(self, win, frame_type: FrameType = FrameType.MIN30):
        if frame_type == FrameType.MIN30:
            return {5: 3e-3}.get(win, 1e-3)
        elif frame_type == FrameType.DAY:
            return {5: 8e-3, 10: 5e-3}.get(win, 3e-3)

    async def scan(self, end: Frame, codes: List=None, profit: float=0.1):
        end = arrow.get(end)

        async def get_code(codes):
            if codes is None:
                return await omicron.cache.sys.lpop(f"scan.scope.{self.name}")
            else:
                if len(codes):
                    return codes.pop()
                else:
                    return None

        while (code := await get_code(codes)) is not None:
            try:
                sec = Security(code)
                result = await self.xtransform(end, sec, profit)

                if result is None:
                    continue

                result.extend((code, sec.display_name, end))
                await self.process_features(result)
            except Exception as e:
                logger.exception(e)

    async def process_features(self, features):
        self.X.append(features)

        features_ = []
        features_.extend(map(float, features[:-3]))
        features_.append(features[-3])
        features_.append(features[-2])
        features_.append(self.format_frame(features[-1]))

        tm = arrow.now().format("YYMMDD")
        await omicron.cache.sys.lpush(f"scan.result.{self.name}.{tm}", json.dumps(features_))

    def format_frame(self, frame):
        if hasattr(frame, "hour") and frame.hour != 0:
            fmt = "YY-MM-DD HH:mm"
        else:
            fmt = "YY-MM-DD"

        return arrow.get(frame).format(fmt)

    def is_bars_valid(self, bars):
        if len(bars) != self.n_minbars:
            return False

        close = bars["close"]

        if np.count_nonzero(np.isfinite(close)) < len(close) * 0.9:
            return False

        return True

    def get_ma_fitlen(self, win:int):
        return {
            5: 7,
            10: 10
        }.get(win, 15)

    async def xtransform(self, end: Frame, sec: Security, profit:float=0.1):
        frame_type = FrameType.MIN30

        features = []
        if end is None:
            end = tf.floor(arrow.now(), frame_type)

        if not hasattr(end, 'hour'):
            end = tf.combine_time(end, 15)
        elif end.hour == 0:
            end = tf.combine_time(end, 15)

        start = tf.shift(end, -self.n_minbars + 1, frame_type)
        bars = await sec.load_bars(start, end, frame_type)

        if not self.is_bars_valid(bars):
            return None

        close = fillna(bars["close"].copy())

        ypreds, _ = predict_by_moving_average(close, 5, 5, self.get_pmae_err_threshold(5, frame_type))

        if ypreds is None:
            return None

        profit_ = ypreds[-1] / close[-1] - 1
        if profit_ < profit:
            return None

        # min30 profit
        features.append(profit_)

        # m30 ma fit line coeffs
        for win in self.ma_min_wins:
            ma = moving_average(close, win)[-self.get_ma_fitlen(win):]

            (a,b,c), pmae = polyfit(ma/ma[0])
            features.extend((a,b, pmae))

        # m30 volume features
        vf = top_volume_direction(bars, 16)
        features.extend(vf)

        # m30 rsi
        rsi = relative_strength_index(close, 6)
        features.extend(rsi[-5:])

        # day level features
        frame_type = FrameType.DAY
        start = tf.shift(end.date(), -self.n_daybars + 1, frame_type)
        bars = await sec.load_bars(start, end.date(), frame_type)
        if not self.is_bars_valid(bars):
            return None

        close = fillna(bars["close"].copy())

        ## 提取各周期均线的拟合系数
        for win in self.ma_day_wins:
            fit_len = self.get_ma_fitlen(win)
            ma = moving_average(close, win)[-fit_len:]
            (a,b,_), pmae = polyfit(ma/ma[0])
            features.extend((a,b, pmae))

        # day level volume features
        vf = top_volume_direction(bars)
        features.extend(vf)

        # day level rsi
        rsi = relative_strength_index(close, 6)
        features.extend(rsi[-5:])

        # day level relation with prev_high
        rh = relation_with_prev_high(close)
        features.extend(rh)

        return features

    def to_dataframe(self, features):
        ma30m_coeffs = np.array([
            (f"a30m{win}", f"b30m{win}", f"e30m{win}") for win in self.ma_min_wins
        ]).flatten().tolist()

        ma1d_coeffs = np.array([
            (f"a1d{win}", f"b1d{win}", f"e1d{win}") for win in self.ma_day_wins
        ]).flatten().tolist()

        cols = [
            "profit",
            *ma30m_coeffs,
            "v30m1",
            "v30m2",
            "v30m3",
            "rsi30m1",
            "rsi30m2",
            "rsi30m3",
            "rsi30m4",
            "rsi30m5",
            *ma1d_coeffs,
            "v1d1",
            "v1d2",
            "v1d3",
            "rsi1d1",
            "rsi1d2",
            "rsi1d3",
            "rsi1d4",
            "rsi1d5",
            "rh1d1",
            "rh1d2",
            "code",
            "name",
            "time"
        ]

        df = pd.DataFrame(features, columns=cols)
        return df










