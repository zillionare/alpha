#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

import arrow
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.models.security import Security
from alpha.core import signal

from alpha.plots.baseplot import BasePlot

logger = logging.getLogger(__name__)


class Glance(BasePlot):
    def __init__(self, code):
        super().__init__()
        self.sec = Security(code)
        
    async def evaluate(self, *args, **kwargs):
        pass

    async def month_trend(self, end:Frame=None):
        end = end or tf.floor(arrow.now())
        start = tf.shift(end, -26, FrameType.MONTH)

        bars = await self.sec.load_bars(start, end, FrameType.MONTH)
        ma5 = signal.moving_average(bars['close'], 5)
        ma10 = signal.moving_average(bars['close'], 10)
        ma20 = signal.moving_average(bars['close'], 20)

        err, (a,b,c),(vx,_) = signal.polyfit(ma5[-7:]/ma5[0])
        if err > 1e-3:
            note = f"月线ma5走势震荡中。"
        else:
            pass
