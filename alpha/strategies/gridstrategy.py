import asyncio
import datetime
import json
import logging
import os
import uuid

import arrow
import cfg4py
import numpy as np
import omicron
from coretypes import FrameType
from omicron import tf
from omicron.models.stock import Stock
from traderclient import TraderClient

from alpha.config import get_config_dir
from alpha.strategies.base import BaseStrategy

cfg = cfg4py.get_instance()
logger = logging.getLogger(__name__)


class GridStrategy(BaseStrategy):
    name = "grid-strategy"
    desc = "网络交易策略。按照最近10天的收盘价格，计算出中枢价格，然后根据中枢价格，在3个标准差以内划分网格，进行交易。"

    def __init__(self, **kwargs):
        super().__init__()
        self.name = "grid-strategy-v0"
        self.token = "aaron-grid-token"

        self.mid_price = 0
        self.delta = 0
        self.mid_price_date = None

        self.buckets = {-6: 0, -5: 0, -4: 0, -3: 0, -2: 0, -1: 0}
        self._principal = kwargs.get("principal", 100000)

        # 划分的网格数
        self.grids = len(self.buckets)
        self.cash_per_grid = self.principal / self.grids
        # 卖出时最少网格跨度，[1, self.grids)
        self.min_span = 3
        self.code = ""

    async def backtest(self, start: datetime.date, end: datetime.date, params: dict):
        """

        Args:
            window: 回测的天数
        """
        assert "code" in params

        self.__dict__.update(params)
        bars = await Stock.get_bars_in_range(self.code, FrameType.DAY, start, end)

        for bar in bars:
            buckets = [i for i, v in self.buckets.items() if v > 0]
            logger.info("回测: %s, 非空令牌: %s", bar["frame"], buckets)
            await self._update_params(self.code, bar["frame"])
            await self.evaluate(self.code, bar)

    async def _update_params(self, code: str, dt: datetime.date):
        if (
            self.mid_price_date is None
            or tf.count_day_frames(self.mid_price_date, dt) >= 5
        ):
            bars = await Stock.get_bars(code, 30, FrameType.DAY, dt)

            if len(bars) < 30:
                raise ValueError("%s 小于30天,无法进行回测", code)

            close = bars["close"]
            self.mid_price = np.max(bars["high"]) / 2 + np.min(bars["low"]) / 2
            # 在3个标准差范围内进行交易
            self.delta = np.std(close) * (3 / self.grids)
            self.mid_price_date = dt
            logger.info("更新参数:%s, 中枢价: %.2f, 标准差: %.2f", dt, self.mid_price, self.delta)

    async def _buy(self, code: str, grid: int, price: float, dt: datetime.date):
        dt = tf.combine_time(dt, 9, 31)
        logger.info("%s 委买%s 价格: %.2f, Grid: %d", dt, code, price, grid)
        result = await self.buy(code, price, 1 / self.grids, order_time=dt)
        if result is not None and result["status"] == 0:
            volume = result["data"]["volume"]
            price = result["data"]["price"]
            logger.info("成功买入%d股, 价格: %.2f, Grid: %d", volume, price, grid)
            self.buckets[grid] = volume

    async def _sell(self, code: str, grid: int, price: float, dt: datetime.date):
        dt = tf.combine_time(dt, 9, 31)
        logger.info("%s 委卖%s, 价格: %.2f, Grid: %d", dt, code, price, grid)
        volume = self.buckets[grid]
        result = await self.sell(code, price, volume, order_time=dt)
        if result is not None and result["status"] == 0:
            volume = sum([item["volume"] for item in result["data"]])
            logger.info("成功卖出%d股, 价格: %.2f, Grid: %d", volume, price, grid)
            self.buckets[grid] -= volume

    async def evaluate(self, code, bar):
        low, high = bar["low"], bar["high"]
        for i in range(1, self.grids + 1):
            if low < self.mid_price - i * self.delta and self.buckets[-i] == 0:
                price = self.mid_price - i * self.delta
                await self._buy(code, -i, price, bar["frame"])
            if high > self.mid_price + i * self.delta:
                for j in range(-self.grids, min(-1, i - self.min_span)):
                    if self.buckets[j]:
                        price = self.mid_price + i * self.delta
                        await self._sell(code, j, price, bar["frame"])
                        break
