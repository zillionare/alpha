"""2021年9月初，中关村"""

import pickle
from alpha.plotting.candlestick import Candlestick
import asyncio
import datetime
import functools
import itertools
import json
import logging
import os
from typing import List, NewType

import arrow
import cfg4py
import fire
import numpy as np
import omicron
import pandas as pd
from arrow import Arrow
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security

from alpha.core.features import (
    fillna,
    moving_average,
    predict_by_moving_average,
    relative_strength_index,
    replace_zero,
)
from alpha.features.volume import top_volume_direction

Frame = NewType("Frame", (datetime.date, datetime.datetime, str, Arrow))

logger = logging.getLogger(__name__)


class Zgc(object):
    """中关村"""

    def __init__(self) -> None:
        self.name = self.__class__.__name__.lower()
        self.ma_day_wins = [5, 10, 20, 60]
        self.ma_min_wins = [5, 10, 20]
        self.nbars = max(max(self.ma_day_wins), max(self.ma_min_wins)) + 20
        self.profit = {FrameType.MIN30: 0.05, FrameType.DAY: 0.15}
        self.vf_threshold = {FrameType.MIN30: 4, FrameType.DAY: 2}

        cols = ["code", "name", "end", "gain_day", "gain_min", "rsi_day", "rsi_min"]
        cols.extend([f"gain_day_{win}" for win in self.ma_day_wins[1:]])
        cols.extend([f"gain_min_{win}" for win in self.ma_min_wins[1:]])
        cols.extend([f"vf_day_{i}" for i in range(3)])
        cols.extend([f"vf_min_{i}" for i in range(3)])

        self.cols = cols
        super().__init__()

    def get_pmae_err_threshold(self, win, frame_type: FrameType = FrameType.MIN30):
        if frame_type == FrameType.MIN30:
            return {5: 3e-3}.get(win, 1e-3)
        elif frame_type == FrameType.DAY:
            return {5: 8e-3, 10: 5e-3}.get(win, 3e-3)

    async def scan(self, end: Frame, codes: List = None):
        """
        刷新
        """
        results = []
        end = tf.floor(arrow.get(end), FrameType.MIN30)
        logger.info("end time is %s", end)

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

                result = await self.check_long_signal(end.date(), sec, FrameType.DAY)
                if result is None:
                    continue

                profits_day, rsi_day, vf_day = result

                result = await self.check_long_signal(end, sec, FrameType.MIN30)
                if result is None:
                    continue

                profits_30, rsi_30, vf_30 = result

                row = [code, sec.display_name, end]
                row.extend((profits_day[0], profits_30[0], rsi_day, rsi_30))
                row.extend(profits_day[1:])
                row.extend(profits_30[1:])
                row.extend(vf_day)
                row.extend(vf_30)

                results.append(row)
            except Exception as e:
                logger.exception(e)
                continue

        if codes is not None:  # single process, called directly
            return pd.DataFrame(results, columns=self.cols)
        else:
            await omicron.cache.sys.set(f"scan.scope.{self.name}", self.dump_results(results))

    def dump_results(self, results):
        data = []
        for row in results:
            tmp = []
            tmp.append(row[0], row[1], self.format_frame(row[2]))
            tmp.extend(map(str, row[3:]))
            data.append(tmp)

        return json.dumps(data)


    async def check_long_signal(
        self, end: Frame, sec: Security, frame_type: FrameType = FrameType.MIN30
    ):
        """
        买入
        """
        start = tf.shift(end, -self.nbars + 1, frame_type)
        bars = await sec.load_bars(start, end, frame_type)

        if (
            bars is None
            or len(bars) != self.nbars
            or np.count_nonzero(np.isfinite(bars["close"])) < self.nbars * 0.9
        ):
            return None

        close = fillna(bars["close"].copy())
        # 还在连续下跌之中
        if np.all(close[-2:] < close[-3:-1]):
            return None

        # 如果存在高开大阴线，则不操作
        if self.is_gap_blackline(bars):
            return None

        # 当前股价依托ma10或者ma20
        # ma10 = moving_average(close, 10)
        # ma20 = moving_average(close, 20)

        # c = close[-1]
        # if (
        #     c / ma10[-1] - 1 > self.bias[frame_type]
        #     and c / ma20[-1] > self.bias[frame_type]
        # ):
        #     return None

        vol_win = 16 if frame_type == FrameType.MIN30 else 10
        vf = top_volume_direction(bars, vol_win)
        indice = np.argsort(np.abs(vf))
        vf = vf[indice]

        # 最大成交量放大效应不足，或者资金在流出
        if vf[-1] < self.vf_threshold.get(frame_type, 2) or np.sum(vf) < 0:
            return None

        profits = self.predict_profit(close, frame_type)
        if profits is None:
            return None

        rsi = relative_strength_index(close, 6)

        return profits, round(rsi[1], 1), map(lambda x: round(x, 1), vf)

    def predict_profit(self, close, frame_type, ylen=5):
        """
        预测盈利
        """
        profits = []
        wins = self.ma_day_wins if frame_type == FrameType.DAY else self.ma_min_wins

        w0 = wins[0]
        for win in wins:
            ypreds_, _ = predict_by_moving_average(
                close, win, ylen, self.get_pmae_err_threshold(win, frame_type)
            )

            if ypreds_ is None:
                profits.append(np.nan)
                continue

            profit = round(ypreds_[-1] / close[-1] - 1, 3)
            profits.append(profit)

            # 还处于下跌当中，或者短期利润不满足要求
            if win == w0:
                if ypreds_[0] < close[-1] or profit < self.profit[frame_type]:
                    return None

        return profits

    def is_gap_blackline(self, bars):
        """最后一周期是否为跳空高开阴线"""
        close = bars["close"][-2:]
        open_ = bars["open"][-2:]

        if open_[1] > close[0] and close[1] < close[0]:
            return True

    async def read_cached_results(self):
        data = await omicron.cache.sys.get(f"scan.result.{self.name}")
        results = json.loads(data)
        return pd.DataFrame(results, columns=self.cols)

    def format_frame(self, frame):
        if hasattr(frame, "hour") and frame.hour != 0:
            fmt = "YY-MM-DD HH:mm"
        else:
            fmt = "YY-MM-DD"

        return arrow.get(frame).format(fmt)

    async def plot_results(self, results:pd.DataFrame, save_to:str=None):
        save_to = save_to or "/tmp/zgc"
        os.makedirs(save_to, exist_ok=True)

        cs = Candlestick({
            "1d": [5, 10, 20, 60, 120],
            "30m": [5, 10, 20, 60]
        })

        for row in results.to_records(index=False):
            code = row["code"]
            end = row["end"]
            profit = row["gain_day"]

            title = f"{code} {self.format_frame(end)} {profit:.0%}"
            await cs.plot(code, end, title=title, save_to=save_to)
