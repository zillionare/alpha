#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging
from typing import Callable

from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security
import numpy as np
from pandas import DataFrame

from alpha.core import signal

logger = logging.getLogger(__name__)

class Huierpu:
    """
    600983,惠而浦，2020/3/20-2020-3-25期间，三次接近前低但不创新低（1.01，1.015，1.017),3.25
    当天拉起（盘中最低探明5日线），涨幅4%，上穿5日及10日均线，次日起连续4个涨停。
    """
    async def fire_long(self, end:Frame, frame_type:FrameType.DAY, win=60, adv=0.03):
        secs = Securities()
        results = []
        for code in secs.choose(['stock']):
        #for code in ['601238.XSHG']:
            sec = Security(code)
            if sec.name.find("ST") != -1 or sec.code.startswith("688"):
                continue

            start = tf.shift(end, -win+1, frame_type)
            bars = await sec.load_bars(start, end, frame_type)
            ilow = np.argmin(bars['low'])
            if ilow > win//2:#创新低及后面的反弹太近，信号不可靠
                continue

            low = bars['low'][ilow]
            last = bars['low'][-5:]
            if np.count_nonzero((last > low) & (last < low * 1.02)) < 3:
                # 对新低的测试不够
                continue

            c1,c0 = bars['close'][-2:]
            # 今天上涨幅度是否大于adv?
            if c0/c1- 1 < adv:
                continue
            # 是否站上5日线10日线？
            ma5 = signal.moving_average(bars['close'], 5)
            ma10 = signal.moving_average(bars['close'], 10)

            if c0 < ma5[-1] or c0 < ma10[-1]: continue

            price_change = await sec.price_change(end, tf.day_shift(end, 5), frame_type)
            print(f"FIRED:{end}\t{code}\t{price_change:.2f}")
            results.append([end, code, price_change])

        return results

    async def scan(self, start: Frame, end: Frame, signal_func: Callable,
                   frame_type: FrameType = FrameType.DAY):
        frames = tf.get_frames(start, end, frame_type)
        results = []
        for frame in frames:
            if frame_type in tf.day_level_frames:
                frame = tf.int2date(frame)
            else:
                frame = tf.int2time(frame)
            result = await signal_func(frame, frame_type=frame_type,win=60,adv=0.0)
            results.extend(result)

        df = DataFrame(data=results, columns=['date','code','pc'])
        df.to_csv("/tmp/hrp.csv")