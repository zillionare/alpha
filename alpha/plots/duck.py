#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

import arrow
import cfg4py
from alpha.core.enums import Events
from omicron.core.timeframe import tf
from omicron.core.types import FrameType, Frame
from omicron.models.security import Security
from pyemit import emit

from alpha.core import features, signal
from alpha.core.monitors import MonitorManager

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()

class DuckPlot:
    """
    8-10个周期内5日均线向上开口，上穿10日线当日，或者预计下一日将实现上穿时，发出信号。月线为
    上述两线支撑。
    """
    err_20 = 3e-3
    war_20 = 3e-2
    def __init__(self, scheduler, trade_time_only=True):
        self.name = self.__class__.__name__
        self.monitor = MonitorManager(scheduler, self, trade_time_only)

    def watch(self, freq=3, **params):
        self.monitor.watch(freq, params)

    async def evaluate(self, code:str, frame_type: FrameType, dt:Frame=None):
        logger.debug("测试%s, 参数:%s %s", code, frame_type, dt)

        win = 10
        sec = Security(code)
        end = dt or arrow.now(tz=cfg.tz).datetime
        start = tf.shift(dt, -29, frame_type)
        bars = await sec.load_bars(start, end, frame_type)

        feat = features.ma_lines_trend(bars, [5,10,20])
        ma5, ma10 = feat["ma5"][0], feat["ma10"][0]

        # 判断是否存在vcross
        vcrossed, (idx0, idx1) = signal.vcross(ma5[-win:], ma10[-win:])
        if not vcrossed:
            return
        else:
            dt1, dt2 = bars[-win:][idx0]['frame'], bars[-win:][idx1]['frame']
            logger.info("%s vcross(5,10): %s, %s", sec, dt1, dt2)

        # 月线要向上且形成支撑作用
        err, a, b, vx, fit_win, war = feat["ma20"][1]
        if err < self.err_20 and war > self.war_20:
            logger.info("%s月线向上，war:%s", code, war)

        await emit.emit(Events.sig_long, {
            "code": code,
            "plot": self.name,
            "desc": f"{sec.display_name}发出老鸭头信号"
        })