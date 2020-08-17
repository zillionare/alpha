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
from pyemit import emit

from alpha.core.enums import Events
from omicron.core.timeframe import tf
from omicron.core.types import FrameType, Frame
from omicron.models.security import Security

from alpha.core import features, signal
from alpha.plots.base import AbstractPlot

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


class LongParallel(AbstractPlot):
    """
    均线多头策略。在均线多头刚刚形成之初（粘合，当天一阳穿多线）发出信号
    """
    max_distance = 1e-2  # 华锦股份， 2020-6-22，5.7e-3
    fitslp_ma20 = 2e-3
    fiterr_ma20 = 3e-3
    results = []

    async def evaluate(self, code: str, frame_type: FrameType, dt: Frame = None,
                       win=15):
        sec = Security(code)
        end = dt or arrow.now(tz=cfg.tz)
        start = tf.shift(tf.floor(end, frame_type), -win - 20, frame_type)
        bars = await sec.load_bars(start, end, frame_type)

        if len(bars) < win + 20:
            return

        # 使用股价重心而不是收盘价来判断走势
        o, c = bars[-1]['open'], bars[-1]['close']

        feat = features.ma_lines_trend(bars, [5, 10, 20])
        ma5, ma10, ma20 = feat["ma5"][0], feat["ma10"][0], feat["ma20"][0]

        if np.any(np.isnan(ma5)):
            return

        mas = np.array([ma5[-1], ma10[-1], ma20[-1]])
        # 起涨点：一阳穿三线来确认
        if not (np.all(o <= mas) and np.all(c >= mas)):
            return

        # 三线粘合：三线距离小于self.max_distance
        distance = self.distance(ma5[:-1], ma10[:-1], ma20[:-1])
        if distance > self.max_distance:
            return

        # 月线要拉直，走平或者向上
        err, (a, b) = signal.polyfit((ma20 / ma20[0]), deg=1)
        if err > self.fiterr_ma20 and a < self.fitslp_ma20:
            return

        logger.info("%s", f"{sec.display_name}\t{distance:.3f}\t{a:.3f}\t{err:.3f}")
        await self.fire("long", code, dt, frame_type=frame_type.value, distance=distance)

    def distance(self, ma5, ma10, ma20):
        tl = min(len(ma5), len(ma10), len(ma20))
        if tl <= 5:
            logger.warning("passing param may be not long enough:%s", tl)
        _ma5, _ma10, _ma20 = ma5[-tl:], ma10[-tl:], ma20[-tl:]

        mean = (_ma5 + _ma10 + _ma20) / 3

        return np.sqrt(np.mean(np.square(_ma5 / mean - 1) +
                               np.square(_ma10 / mean - 1) +
                               np.square(_ma20 / mean - 1)
                               ))

    async def copy(self, code: str, frame_type: FrameType, start, stop):
        n = tf.count_frames(start, stop, frame_type)
        bars = await self.get_bars(code, n+20, frame_type, stop)

        feat = features.ma_lines_trend(bars, [5, 10, 20])
        ma5, ma10, ma20 = feat["ma5"][0], feat["ma10"][0], feat["ma20"][0]

        self.max_distance = self.distance(ma5, ma10, ma20)
        err, (a,b) = signal.polyfit(ma20, deg=1)
        self.fitslp_ma20 = a
        self.fiterr_ma20 = err
