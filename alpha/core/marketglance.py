#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import json
import logging

import arrow
from alpha.core.enums import Events
from omicron.core.timeframe import tf
from omicron.core.triggers import FrameTrigger
from omicron.core.types import FrameType
from omicron.dal import cache
from omicron.models.securities import Securities
import cfg4py
import numpy as np
from omicron.models.security import Security
from pyemit import emit

cfg = cfg4py.get_instance()

logger = logging.getLogger(__name__)

class MarketGlance:
    def __init__(self, scheduler):
        self.scheduler = scheduler

        self.price_change_history = []

    async def start(self):
        trigger = FrameTrigger(FrameType.MIN30)
        self.scheduler.add_job(self.distribution, trigger)
        emit.register(Events.sig_trade, self.on_plot_report)

    async def distribution(self):
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

        self.price_change_history.append((zt, dt, cuts))
        if len(self.price_change_history) == 8:
            self.price_change_history.pop(0)

        now = arrow.now(tz=cfg.tz)
        if now.hour >= 15:
            dt = tf.date2int(now)
            await cache.sys.hset(f"glance{dt}", "distribution", json.dumps({
                "zt": zt,
                "dt": dt,
                "cuts": cuts
            }))

        return zt, dt, cuts

    async def report(self):
        pass

    async def on_plot_report(self, msg):
        plot = msg.get("plot")
        code = msg.get("code")


