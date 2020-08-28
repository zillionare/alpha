#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import json
import logging

import arrow
import cfg4py
from omicron.core.timeframe import tf
from omicron.core.types import FrameType, Frame
from omicron.dal import cache
from omicron.models.security import Security
from pyemit import emit

from alpha.core import signal
from alpha.core.enums import Events

cfg = cfg4py.get_instance()
logger = logging.getLogger(__name__)


def create_monitor(name):
    if name == 'maline':
        return TouchMaLine()
    elif name == 'line':
        return ExtendLine()
    elif name == 'fix_price':
        return FixPrice()
    elif name == 'momentum':
        return Momentum()


class BaseMonitor:
    def __init__(self, name: str, display_name: str):
        self.name = name
        self.display_name = display_name

    async def get_bars(self, code: str, n: int, frame_type: FrameType,
                       end_dt: Frame = None):
        end_dt = end_dt or arrow.now(tz=cfg.tz)

        sec = Security(code)
        start = tf.shift(tf.floor(end_dt, frame_type), -n + 1, frame_type)
        return await sec.load_bars(start, end_dt, frame_type)

    async def fire(self, flag: str, code: str, fire_on: Frame,
                   frame_type: FrameType,
                   **kwargs):
        convert = tf.date2int if frame_type in tf.day_level_frames else tf.time2int
        fire_on = convert(fire_on)
        frame = frame_type.value

        sec = Security(code)
        logger.info("%s fired %s on (%s,%s,%s)",
                    self.name, flag, sec.display_name, fire_on, frame)
        event = Events.sig_long if flag == "long" else Events.sig_short

        kwargs.update({
            "monitor":    self.name,
            "name":       self.display_name,
            "code":       code,
            "flag":       flag,
            "fire_on":    fire_on,
            "frame_type": frame_type.value
        })
        await emit.emit(event, kwargs)
        d = {
            f"{code}:{fire_on}": json.dumps(kwargs)
        }
        await cache.sys.hmset_dict(f"monitors.{self.name}.signals", d)


class TouchMaLine(BaseMonitor):
    def __init__(self):
        super().__init__('maline', "均线支撑/压力")

    async def evaluate(self, code: str, frame_type: str, flag: str, win: int,
                       slip: float = 0.015):
        fit_win = 7
        frame_type = FrameType(frame_type)
        bars = await self.get_bars(code, fit_win + win, frame_type)
        ma = signal.moving_average(bars['close'], win)

        c0 = bars[-1]['close']
        if abs(c0 / ma[-1] - 1) <= slip:
            await self.fire(flag, code, bars[-1]['frame'], frame_type,slip=slip,win=win)


class FixPrice(BaseMonitor):
    def __init__(self):
        super().__init__('fix_price', "指定价格")

    async def evaluate(self, code: str, frame_type: str, flag: str, win: int,
                       price: float,
                       slip: float = 0.015):
        frame_type = FrameType(frame_type)
        bars = await self.get_bars(code, 1, frame_type)
        if abs(bars[-1]['close'] / price - 1) <= slip:
            await self.fire(flag, code, bars[-1]['frame'], frame_type)


class ExtendLine(BaseMonitor):
    def __init__(self):
        super().__init__('line', '延长线')

    async def evaluate(self, code: str, frame_type: str, flag: str, win: int,
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
            await self.fire(flag, code, bars[-1]['frame'], frame_type)


class Momentum(BaseMonitor):
    """
    一系列基于动量的分析方法。
    比如，对于2020-8-7前后的大盘，5日均线呈抛物线趋势，当天达到最高点；月线则于7月31日达到顶点
    当前下降中；10日线则还在上升中。

    2020-8-14 15:00, 大盘 30m ma5线3e-4、ma20线1e-4
    对个股可能使用1e-3？
    """

    mom = 1e-4
    err = 3e-3
    fit_win = 7

    def __init__(self):
        super().__init__('momentum', '动量转折')

    async def evaluate(self, code: str, frame_type: str = '30m',
                       dt: str = None,
                       win=5,
                       flag='long',
                       mom=None,
                       fit_win=7):
        stop = arrow.get(dt, tzinfo=cfg.tz) if dt else arrow.now(tz=cfg.tz)
        frame_type = FrameType(frame_type)
        mom = mom or self.mom
        fit_win = fit_win or self.fit_win

        bars = await self.get_bars(code, win + fit_win, frame_type, stop)

        ma = signal.moving_average(bars['close'], win)
        _ma = ma[-fit_win:]
        err, (a, b, c), (vx, _) = signal.polyfit(_ma / _ma[0])
        if err > self.err:
            return

        sec = Security(code)
        vx = int(vx)
        logger.info("%s: a=%s, vx=%s, mom=%s", sec.display_name, a, vx, mom)
        if a > mom and 0 < vx < fit_win - 2 and flag in ['both', 'long']:
            # 先降后升，上升2周期
            await self.fire('long', code, stop, frame_type=frame_type, vx=vx, mom=a)
        if a < -mom and vx <= fit_win - 2 and flag in ['both', 'short']:
            # 先升后降，离顶2周期
            await self.fire('short', code, stop, frame_type=frame_type, mom=a, vx=vx)
