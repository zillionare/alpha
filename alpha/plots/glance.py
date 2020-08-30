#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

import arrow
import cfg4py
import numpy as np
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security

from alpha.core import signal
from alpha.plots.baseplot import BasePlot

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


class Glance(BasePlot):
    def __init__(self):
        super().__init__("概览")

    async def evaluate(self, *args, **kwargs):
        pass

    async def overview(self):
        pass

    async def price_change(self):
        # 涨停、跌停
        zt, dt = 0, 0
        codes = Securities().choose(['stock'])
        end = arrow.now(cfg.tz).floor('minute').datetime
        pct = []
        async for code, bars in Security.load_bars_batch(codes, end, 2, FrameType.DAY):
            c1, c0 = bars[-2:]['close']
            if (c0 + 0.01) / c1 - 1 > 0.1:
                zt += 1
            if (c0 - 0.01) / c1 - 1 < -0.1:
                dt += 1

            pct.append(c0 / c1 - 1)

        # 分布
        cuts = np.histogram(pct, bins=[-0.2, -0.1, -0.07, -0.03, 0, 0.03,
                                       0.07, 0.1, 0.2])
        return zt, dt, cuts

    async def momentum(self, code: str, frame_type: FrameType):
        end = arrow.now(tz=cfg.tz).datetime
        start = tf.shift(end, -26, frame_type)
        bars = await Security(code).load_bars(start, end, frame_type)

        score = 0
        details = []

        # ma5
        ma5 = signal.moving_average(bars['close'], 5)
        err, (a, b, c), (vx, _) = signal.polyfit(ma5[-7:] / ma5[-7])
        err_baseline = 3e-3 if frame_type == FrameType.MIN30 else 6e-3
        if err < err_baseline:
            p = np.poly1d((a, b, c))
            y = p(9) / p(6) - 1
            details.append((y, vx, a, b, err))
            score += y
        else:
            details.append((None, None, None, None, err))

        # ma10
        ma10 = signal.moving_average(bars['close'], 10)
        err, (a, b, c), (vx, _) = signal.polyfit(ma10[-7:] / ma10[-7])
        err_baseline = 3e-3 if frame_type == FrameType.MIN30 else 6e-3
        if err < err_baseline:
            p = np.poly1d((a, b, c))
            y = p(9) / p(6) - 1
            details.append((y, vx, a, b, err))
            score += y * 2
        else:
            details.append((None, None, None, None, err))

        # ma20
        ma20 = signal.moving_average(bars['close'], 20)
        err, (a, b, c), (vx, _) = signal.polyfit(ma20[-7:] / ma20[-7])
        err_baseline = 3e-3 if frame_type == FrameType.MIN30 else 6e-3
        if err < err_baseline:
            p = np.poly1d((a, b, c))
            y = p(9) / p(6) - 1
            details.append((y, vx, a, b, err))
            score += y * 5
        else:
            details.append((None, None, None, None, err))

        return score, details


    async def month_trend(self, end: Frame = None):
        end = end or tf.floor(arrow.now())
        start = tf.shift(end, -26, FrameType.MONTH)

        bars = await self.sec.load_bars(start, end, FrameType.MONTH)
        ma5 = signal.moving_average(bars['close'], 5)
        ma10 = signal.moving_average(bars['close'], 10)
        ma20 = signal.moving_average(bars['close'], 20)

        err, (a, b, c), (vx, _) = signal.polyfit(ma5[-7:] / ma5[-7])
        if err > 1e-3:
            note = f"月线ma5走势震荡中。"
        else:
            pass
