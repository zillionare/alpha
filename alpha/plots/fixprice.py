#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging
from typing import Union

from omicron.core.types import FrameType

from alpha.plots.baseplot import BasePlot

logger = logging.getLogger(__name__)


class FixPrice(BasePlot):
    """
    经过人工分析，当股价运行到某一位置时触发交易信号。比如大盘指数的整数关口。
    """

    def __init__(self):
        super().__init__("指定价格")

    async def evaluate(self, code: str, frame_type: Union[str, FrameType], flag: str,
                       win: int,
                       price: float,
                       slip: float = 0.015):
        frame_type = FrameType(frame_type)
        bars = await self.get_bars(code, 1, frame_type)
        if abs(bars[-1]['close'] / price - 1) <= slip:
            await self.fire_trade_signal(flag, code, bars[-1]['frame'], frame_type)
