#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

import numpy as np
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security
from pyemit import emit

from alpha.core import signal
from alpha.core.enums import Events
from alpha.plots.baseplot import BasePlot

logger = logging.getLogger(__name__)


class NinePlot(BasePlot):
    def __init__(self):
        super().__init__()
        self.ref_lines = {}

    async def evaluate(self, code, frame_type, flag, ma_win: int = 20, slp=1e-2):
        """
        当股价回归到指定的均线上时，发出信号
        Args:
            code:
            frame_type:
            flag:
            ma:需要监控的均线

        Returns:

        """
        bars = await self.get_bars(code, ma_win + 10, frame_type)
        ma = signal.moving_average(bars['close'], ma_win)

        err, (a, b, c), (vx, _) = signal.polyfit(ma[-7:] / ma[-7])
        p = np.poly1d((a, b, c))
        if (p(11) / p(6) - 1) / 5 < slp:
            return

        sec = Security(code)
        alarm = f"{sec.display_name}触达{ma_win}均线。"
        await emit.emit(Events.sig_long, {"alarm": alarm})

    async def copy(self, code: str, frame_type: FrameType, end: Frame, ma_win=5):
        fit_win = 7

        sec = Security(code)
        start = tf.shift(end, -(ma_win + fit_win), frame_type)
        bars = await sec.load_bars(start, end, frame_type)
        ma = signal.moving_average(bars['close'], ma_win)
        err, (a, b, c), (vx, _) = signal.polyfit(ma[-fit_win:] / ma[-fit_win])
        p = np.poly1d((a, b, c))
        slp3 = p(fit_win + 2) / p(fit_win - 1) - 1
        print(f"{sec.display_name}({code})\t{err:.4f}\t{a:.4f}\t{b:.4f}\t{vx:.1f}\
        \t{slp3:.2f}")
        self.ref_lines[f"ma{ma_win}"] = {
            "err":  err,
            "coef": (a, b),
            "vx":   vx,
            "slp3": slp3
        }

    async def scan_1(self, ma_win: int, frame_type: FrameType, a: float = None,
                   b: float = None, err=1e-3, end: Frame = None):
        """
        在所有股票中，寻找指定均线强于拟合均线(a,b,1)的，如果当前收盘价在均线附近，且近期存在
        大阳线，则发出信号
        Returns:

        """
        if a is None:
            err = self.ref_lines[f"ma{ma_win}"].get("err")
            a, b = self.ref_lines[f"ma{ma_win}"].get("coef")

        fit_win = 7
        secs = Securities()
        p = np.poly1d((a, b, 1.0))
        slp3 = p(fit_win - 1 + 3) / p(fit_win - 1) - 1
        count = 0
        for i, code in enumerate(secs.choose(['stock'])):
            if (i + 1) % 500 == 0:
                logger.debug("handled %s", i+1)
            sec = Security(code)

            bars = await self.get_bars(code, fit_win + 19, frame_type, end)
            ma = signal.moving_average(bars['close'], ma_win)
            err_, (a_, b_, c_), (vx_, _) = signal.polyfit(ma[-fit_win:] / ma[-fit_win])
            if err_ > err:
                continue

            #p_ = np.poly1d((a_,b_,1.0))
            # 如果abs(b) < fit_win * a，曲线（在x不超过fit_win的地方）接近于直线，此时应该比较b
            t5, t10, t20 = False, None, None
            #slp3_5 = p_(fit_win+2)/p_(fit_win-1) - 1
            t5 = a_ >= a*0.99 and fit_win+1 >= vx_ >= fit_win - 2
            if t5:
                print(f"{sec.display_name},{vx_:.1f}")

            # 10日线、20日线不能向下
            # ma10 = signal.moving_average(bars['close'], 10)
            # err_, (a_,b_,_), (vx_, _) = signal.polyfit(ma10[-fit_win:] / ma10[-fit_win])
            # #p_ = np.poly1d((a_,b_,1.0))
            # #slp3_10 = p_(fit_win+2)/p_(fit_win-1) - 1
            # if err_ > 3e-3:
            #     t10 = None
            # elif a_ > 1e-4 or b > 10 * abs(a_):
            #     t10 = True
            # else:
            #     t10 = False
            #
            # ma20 = signal.moving_average(bars['close'], 20)
            # err_, (a_,b_,_), (vx_, _) = signal.polyfit(ma20[-fit_win:] / ma20[-fit_win])
            # #p_ = np.poly1d((a_,b_,1.0))
            # #slp3_20 = p_(fit_win+2)/p_(fit_win-1) - 1
            # if err_ > 1e-3:
            #     t20 = None
            # elif a_ > 1e-4 or b > 10 * abs(a_):
            #     t20 = True
            # else:
            #     t20 = False
            #
            # if t5 and (all([t10,t20]) or
            #            (t10 is None and t20) or
            #            (t20 is None and t10)):
            #     print(f"{sec.display_name}, {[t5, t10, t20]}")

    async def scan(self,stop:Frame=None):
        start = tf.shift(stop, -26, FrameType.WEEK)
        ERR = {
            5:  0.008,
            10: 0.004,
            20: 0.004
        }

        for code in Securities().choose(['stock']):
        #for code in ['002150.XSHE']:
            sec = Security(code)
            bars = await sec.load_bars(start, stop, FrameType.WEEK)
            if bars[-1]['frame'] != stop:
                raise ValueError("")

            t1, t2, t3 = False, False, False
            params = []
            for win in [5, 10, 20]:
                ma = signal.moving_average(bars['close'], win)
                err, (a, b, c), (vx, _) = signal.polyfit(ma[-7:] / ma[-7])
                if err > ERR[win]:
                    continue

                p = np.poly1d((a, b, c))
                slp3 = round(p(9) / p(6) - 1, 2)

                params.append(np.round([slp3, a, b], 4))
                if win == 5:
                    t1 = slp3 >= 0.03 and a > 0.005
                if win == 10:
                    t2 = slp3 >= 0.02 and (b > abs(10 * a) or a > 0.0005)
                if win == 20:
                    t3 = slp3 >= -1e-6 and a >= 0

            if all([t1, t2, t3]):
                print(sec.display_name, params)
