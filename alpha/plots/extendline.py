#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging
from typing import Union

import arrow
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType

from alpha.plots.baseplot import BasePlot

logger = logging.getLogger(__name__)


class ExtendLine(BasePlot):
    """
    当股价运行到箱体的上沿或者下沿时，发出信号。箱体的上沿和下沿通过指定两点的延长线来确定。
    """

    def __init__(self):
        super().__init__('延长线')

    async def evaluate(self, code: str, frame_type: Union[FrameType, str], flag: str,
                       win: int,
                       c1: float, d1: Frame,
                       c2: float, d2: Frame,
                       slip: float = 0.015):
        frame_type = FrameType(frame_type)
        n1 = tf.count_frames(d1, d2, frame_type)
        slp = (c2 - c1) / n1
        n2 = tf.count_frames(d2, tf.floor(arrow.now(), frame_type),
                             frame_type)
        c_ = c2 + slp * n2
        bars = await self.get_bars(code, 1, frame_type)
        if abs(c_ / bars[-1]['close'] - 1) <= slip:
            await self.fire_trade_signal(flag, code, bars[-1]['frame'], frame_type)
