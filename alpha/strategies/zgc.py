"""2021年9月初，中关村"""

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
        self.ma_day_wins = [5, 10, 20, 60]
        self.ma_min_wins = [5, 10, 20, 60]
        self.nbars = 80
        self.profit = {FrameType.MIN30: 0.05, FrameType.DAY: 0.25}
        self.bias = {FrameType.MIN30: 0.05, FrameType.DAY: 0.15}
        super().__init__()

    def get_pmae_err_threshold(self, win, frame_type: FrameType = FrameType.MIN30):
        if frame_type == FrameType.MIN30:
            return {5: 3e-3, 10: 1e-3, 20: 1e-3}.get(win, 1e-4)
        elif frame_type == FrameType.DAY:
            return {5: 8e-3, 10: 5e-3, 20: 3e-3}.get(win, 3e-3)

    async def scan(self, codes: List = None, end=None):
        """
        刷新
        """
        results = []
        end = tf.floor(arrow.get(end or arrow.now()), FrameType.MIN30)

        async def get_code(codes):
            if codes is None:
                return await omicron.cache.sys.lpop("scan.scope.Zgc")
            else:
                return codes.pop()

        while (code:=await get_code(codes)) is not None:
            try:
                sec = Security(code)

                result = await self.check_long_signal(end, sec, FrameType.DAY)
                if result is None:
                    continue

                ypred_day, rsi_day = result
                result = await self.check_long_signal(end, sec, FrameType.MIN30)
                if result is None:
                    continue

                ypred_30, rsi_30 = result

                print(
                    f"{sec.display_name:<8}\t{ypred_day:.0%}\t{rsi_day:.0f}\t{ypred_30:.0%}\t{rsi_30:.0f}"
                )

                results.append((sec.display_name, ypred_day, rsi_day, ypred_30, rsi_30, ))
            except Exception as e:
                logger.exception(e)
                continue

        if codes is not None: # single process, called directly
            return pd.DataFrame(
                results, columns=["name", "profit_day", "rsi_day", "profit_min", "rsi_min"]
            )
        else:
            for r in results:
                await omicron.cache.sys.lpush("scan.result.Zgc", json.dumps([r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4])]))

    async def check_long_signal(
        self, end: Frame, sec: Security, frame_type: FrameType = FrameType.MIN30
    ):
        """
        买入
        """
        nbars = 100
        start = tf.shift(end, -nbars + 1, frame_type)
        bars = await sec.load_bars(start, end, frame_type)

        if (
            bars is None
            or len(bars) != nbars
            or np.count_nonzero(np.isfinite(bars["close"])) < nbars * 0.9
        ):
            return None

        open_ = fillna(bars["open"].copy())
        close = fillna(bars["close"].copy())

        # 如果存在高开大阴线，则不操作
        if self.is_gap_blackline(bars):
            return None

        # # 最后两周期下跌
        # if np.all(close[-2:] < close[-3:-1]):
        #     return None

        # 当前股价依托ma10或者ma20
        ma10 = moving_average(close, 10)
        ma20 = moving_average(close, 20)

        c = close[-1]
        if (
            c / ma10[-1] - 1 > self.bias[frame_type]
            and c / ma20[-1] > self.bias[frame_type]
        ):
            return None

        vol_win = 16 if frame_type == FrameType.MIN30 else 10
        vf = top_volume_direction(bars, vol_win)
        indice = np.argsort(np.abs(vf))
        vf = vf[indice]

        # 最大成交量方向为下跌
        if vf[-1] < 0:
            return None

        ypred = self.predict_profit(close, frame_type)
        if ypred is None or ypred < self.profit[frame_type]:
            return None

        rsi = relative_strength_index(close, 14)
        return ypred, rsi[1]

    def predict_profit(self, close, frame_type, ylen=5):
        """
        预测盈利
        """
        ypred = 0
        for win in self.ma_min_wins:
            _ypreds, _ = predict_by_moving_average(
                close, win, ylen, self.get_pmae_err_threshold(win, frame_type)
            )

            if _ypreds is None:
                continue

            # 如果长线明显看空，则不操作
            short = 0.98 if frame_type == FrameType.MIN30 else 0.95
            if _ypreds[-1] < close[-1] * short and win in [10, 20, 60]:
                return None

            ypred = max(ypred, max(_ypreds))

        return ypred / close[-1] - 1

    def is_gap_blackline(self, bars):
        """最后一周期是否为跳空高开阴线"""
        close = bars["close"][-2:]
        open_ = bars["open"][-2:]

        if open_[1] > close[0] and close[1] < close[0]:
            return True

    async def read_cached_results(self):
        data = await omicron.cache.sys.lrange("scan.result.Zgc", 0, -1)

        results = []
        for d in data:
            results.append(json.loads(d))
        return pd.DataFrame(results, columns=["name", "profit_day", "rsi_day", "profit_min", "rsi_min"])
