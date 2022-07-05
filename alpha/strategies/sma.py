import asyncio
import datetime
import logging

import numpy as np
import traderclient
from coretypes import FrameType
from omicron import moving_average, tf
from omicron.models.stock import Stock

from alpha.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class SMAStrategy(BaseStrategy):
    """traditionaly ma cross strategy

    Args:
        BaseStrategy : _description_
    """

    name = "sma"
    alias = "SMA strategy"
    desc = """双均线策略，5日线上穿10日线买入，反之卖出。参数:
        {
            "code": "000001.XSHE",
            "frame_type": "1d"
        }
    """
    version = "v1"

    def __init__(self):
        super().__init__()

    def check_required_params(self, params: dict):
        if "code" not in params:
            params["code"] = "000001.XSHE"
        if "frame_type" not in params:
            params["frame_type"] = "1d"

        try:
            ft = FrameType(params.get("frame_type"))
        except Exception:
            raise ValueError(f"frame_type {ft} is invalid")

    async def backtest(self, start: datetime.date, end: datetime.date):
        code = self._bt.code
        frame_type = FrameType(self._bt.frame_type)

        bars = await Stock.get_bars_in_range(code, frame_type, start, end)
        ma5 = moving_average(bars["close"], 5)
        ma10 = moving_average(bars["close"], 10)

        for i in range(1, len(bars) - 1):
            await self.update_progress(bars[i]["frame"])
            if ma10[i] is None:
                continue

            p10, n10 = ma10[i - 1 : i + 1]
            p5, n5 = ma5[i - 1 : i + 1]

            if np.isnan(p10):
                continue

            try:
                if p5 <= p10 and n5 > n10:
                    shares = self.principal / bars[i]["close"]
                    await self.buy(code, shares, bars["frame"][i])
                    continue

                if p5 >= p10 and n5 < n10:
                    await self.sell(code, 1, bars["frame"][i])
            except traderclient.errors.TradeError as e:
                logger.exception("backtest error")
                logger.exception(e)
