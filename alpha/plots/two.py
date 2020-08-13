#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging
from typing import Callable

import arrow
import cfg4py
import numpy as np
from omicron.core.timeframe import tf
from omicron.core.types import FrameType, Frame
from omicron.models.securities import Securities
from omicron.models.security import Security
from pandas import DataFrame

from alpha.core import signal

logger = logging.getLogger(__name__)

cfg = cfg4py.get_instance()


class Two:
    """
    样本： 西部资源 600139， 2020-08-04 10：00：00

    1. 股价高于所有均线
    2. 30分钟线的ma5, ma10, ma60粘合(overlap_win周期）后再多头
    3. ma60向上
    4. 日线级别前几日(n<7)必须有放量大阳
    """

    async def fire_long(self, end: Frame = None, overlap_win=10,
                        frame_type: FrameType = FrameType.MIN30):
        """
        寻找开多仓信号
        Args:

        Returns:

        """
        result = []
        end = end or arrow.now().datetime
        secs = Securities()
        for code in secs.choose(['stock']):
        #for code in ['600139.XSHG']:
            try:
                sec = Security(code)
                start = tf.shift(end, -(60 + overlap_win - 1), frame_type)
                bars = await sec.load_bars(start, end, frame_type)

                mas = {}
                for win in [5, 10, 20, 60]:
                    ma = signal.moving_average(bars['close'], win)
                    mas[f"{win}"] = ma

                # 收盘价高于各均线值
                c1, c0 = bars['close'][-2:]
                t1 = c0 > mas["5"][-1] and c0 > mas["10"][-1] and c0 > mas["20"][-1] \
                     and c0 > mas["60"][-1]

                # 60均线斜率向上
                slope_60, err = signal.slope(mas["60"][-10:])
                if err is None or err > 5e-4:
                    continue

                t2 = slope_60 >= 5e-4

                # 均线粘合
                diff = np.abs(mas["5"][-6:-1] - mas["10"][-6:-1]) / mas["10"][-6:-1]
                overlap_5_10 = np.count_nonzero(diff < 5e-3)
                t3 = overlap_5_10 > 3

                diff = np.abs(mas["10"][-10:] - mas["60"][-10:]) / mas["60"][-10:]
                overlap_10_60 = np.count_nonzero(diff < 5e-3)
                t4 = overlap_10_60 > 5

                price_change = await sec.price_change(end,
                                                      tf.shift(end, 8, frame_type),
                                                      frame_type)
                result.append([end, code, t1, t2, t3, t4, slope_60, price_change, True])

                if t1 and t2 and t3 and t4:
                    print("FIRED:", [end, code, t1, t2, t3, t4, slope_60,
                                     price_change, True])

            except Exception as e:
                pass

        return result

    async def scan(self, start: Frame, end: Frame, signal_func: Callable,
                   frame_type: FrameType = FrameType.DAY):
        frames = tf.get_frames(start, end, frame_type)
        results = []
        for frame in frames:
            if frame_type in tf.day_level_frames:
                frame = tf.int2date(frame)
            else:
                frame = tf.int2time(frame)
            result = await signal_func(frame, frame_type=frame_type)
            results.extend(result)

        df = DataFrame(data=results,
                       columns=['date', 'code', 't1', 't2','t3','t4', 'slope_60',
                                'pct', "fired"])
        df.to_csv("/tmp/two.csv")
