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

from alpha.core import signal

logger = logging.getLogger(__name__)


class Momentum:
    """
    一系列基于动量的分析方法。
    比如，对于2020-8-7前后的大盘，5日均线呈抛物线趋势，当天达到最高点；月线则于7月31日达到顶点
    当前牌下降中；10日线则还在上升中。
    """
    async def glance(self, code):
        """
        1. 在5,10,20,60,120,250均线中找出当前最接近的位置，使用(5,10)这样的元组来表示
        2. 各均线走势
        Returns:

        """
        now = arrow.now()
        stop_date = now.date()
        start_date = tf.day_shift(stop_date, -269)
        bars = await Security(code).load_bars(start_date, stop_date, FrameType.DAY)
        c0 = bars[-1]['close']
        mas = {}
        for fit_win, win in zip([7,7,10,10,10,10],[5,10,20,60,120,250]):
            ma = signal.moving_average(bars['close'], win)
            err, (a,b,c), (x,_) = signal.polyfit(ma[-fit_win:]/ma[-fit_win])
            mas[f"{win}"] = {
                "ma": ma,
                "fit_err": err,
                "fit_a":a,
                "fit_b": b,
                "vx": x
            }

        ordered = sorted(mas.items(), key=lambda x:x[1]["ma"])
        if c0 < ordered[0][1]['ma']:
            pos = "under_all_ma"
        for i, (key, value) in enumerate(ordered):
            if c0 > value["ma"]:
                if i < len(ordered) - 1:
                    pos = f"({key}-{ordered[i+1][0]})"
                else:
                    pos = "above_all_ma"
                break




    async def ma_turning_point(self,
                               code: str,
                               frame_type: FrameType,
                               end:Frame=None,
                               ma_win: int = 5,
                               fit_win: int = 7,
                               max_err: float = 1e-3):
        """

        Args:
            code: the stock code
            ts:
            ma_win:
            fit_win:
            max_err:

        Returns:

        """
        tslen = ma_win + fit_win - 1
        end = end or arrow.now()
        start = tf.shift(tf.floor(end), -tslen, frame_type)

        bars = await Security(code).load_bars(start, end, frame_type)
        ts = bars['close']
        ma = signal.moving_average(ts, ma_win)
        err, (a, b, c), (x, y) = signal.polyfit(ma[-fit_win:] / ma[fit_win])

        if err < max_err and abs(fit_win - x - 1) <= 2:
            if abs(a) > 5e-4:
                flag = 2 if a > 0 else -2 # 加速上升或者加速下降
            else:
                flag = 1 if b > 0 else -1 # 匀速上升或者匀速下降
            logger.info("SIGNAL[ma_turning_point]:%s,ETA: (%s) frames", code, x)
            await emit.emit("/alpha/signals/ma_turning_point", {
                "code": code,
                "frame_type": frame_type,
                "vx": fit_win -round(x) -1,
                "flag": flag,
                "coef": (a,b,c)
            })
