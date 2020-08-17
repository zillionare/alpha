#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

import arrow
from omicron.core.timeframe import tf
from omicron.core.types import FrameType, Frame
from omicron.models.security import Security
from pyemit import emit
import numpy as np
import cfg4py
from alpha.core import signal
from alpha.plots.base import AbstractPlot

logger = logging.getLogger(__name__)

cfg = cfg4py.get_instance()
class Momentum(AbstractPlot):
    """
    一系列基于动量的分析方法。
    比如，对于2020-8-7前后的大盘，5日均线呈抛物线趋势，当天达到最高点；月线则于7月31日达到顶点
    当前牌下降中；10日线则还在上升中。
    """
    mom = 1e-3
    async def evaluate(self, code: str, frame_type:str = '30m',
                       dt: str = None,
                       win=10,
                       flag='long',
                       mom=None):
        stop = arrow.get(dt, tzinfo=cfg.tz)
        frame_type = FrameType(frame_type)
        mom = mom or self.mom

        fit_win = 7 if win < 10 else 10
        bars = await self.get_bars(code, win + fit_win, frame_type, stop)

        ma = signal.moving_average(bars['close'], win)
        _ma = ma[-fit_win:]
        err, (a, b, c), (vx, _) = signal.polyfit(_ma/_ma[0])
        if err > 3e-3:
            return

        sec = Security(code)
        if a > mom and 0<vx < win - 2 and flag == 'long':  # 先降后升，上升2周期
            desc = f"{self.name}策略看多信号：股票{sec.display_name},上涨两周期。"
            await self.fire('long', code, stop, frame_type=frame_type, vx=vx, mom=a,
                            desc=desc)
        if a < mom and vx >= win - 3 and flag == 'short': # 先升后降，离顶2周期
            desc = f"{self.name}策略看空信号：股票{sec.display_name},离顶{vx-win}周期"
            await self.fire('short', code, stop, frame_type=frame_type,mom=a,vx=vx,
                            desc=desc)
