#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import datetime
import logging
from typing import List, Union

import arrow
import cfg4py
import numpy as np
from omicron.core.timeframe import tf
from omicron.core.types import FrameType, Frame
from omicron.models.securities import Securities
from omicron.models.security import Security

from alpha.core import signal
from alpha.plots.baseplot import BasePlot

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


class Momentum(BasePlot):
    """
    通过动量方法来寻找股票的买入机会，并自动增加买入点监控

    江龙船艇， 2020/8/27
    """

    def __init__(self):
        super().__init__("动量策略")
        self.fit_win = 7
        self.baselines = {
            "up_limit": {
                "30m": 0.01,
                "1d":  0.035
            },
            "ma5":      {
                "30m": {
                    "err": 3e-3,
                    "a":   3e-4,
                    "vx":  (3, 6),
                    "y":   3e-2
                },
                "1d":  {
                    "err": 6e-3,
                    "a":   3e-3,
                    "vx":  (3, 6),
                    "y":   5e-2
                }
            },
            "ma10":     {
                "1d": {
                    "err": 3e-3
                },
                "30m": {
                    "err": 3e-4,
                    "a": 1e-4,
                    "b": 1e-3
                }
            },
            "ma20":     {
                "1d": {
                    "err": 3e-3
                },
                "30m": {
                    "err": 3e-4,
                    "a": 1e-4,
                    "b": 1e-3
                }
            }
        }

    async def pooling(self, frame_type: FrameType = FrameType.DAY, end: Frame = None,
                      codes: List[str] = None):
        if end is None:
            end = arrow.now(cfg.tz).datetime

        assert type(end) in (datetime.date, datetime.datetime)

        codes = codes or Securities().choose(['stock'])
        async for code, bars in Security.load_bars_batch(codes, end, 11, frame_type):
            ma5 = signal.moving_average(bars['close'], 5)
            if len(ma5) < 7:
                continue

            err, (a, b, c), (vx, _) = signal.polyfit(ma5[-7:] / ma5[-7])
            # 无法拟合，或者动能不足
            if err > self.baseline("err", frame_type, "ma5") or a < self.baseline(
                    "a", frame_type, "ma5"):
                continue

            # 时间周期上应该是信号刚出现，还在窗口期内
            vx_range = self.baseline("vx", frame_type, "ma5")
            if not vx_range[0] < vx < vx_range[1]:
                continue

            c1, c0 = bars[-2:]['close']
            cmin = min(bars['close'][-7:-1])

            # 还处在下跌状态、或者涨太多
            if c0 <= cmin or (c0 / c1 - 1) > self.baseline("up_limit", frame_type):
                continue

            p = np.poly1d((a, b, c))
            y = p(9) / p(6) - 1
            # 如果预测未来三周期ma5上涨幅度不够
            if y < self.baseline("y", frame_type, "ma5"):
                continue

            sec = Security(code)

            if frame_type in tf.day_level_frames:
                start = tf.shift(tf.floor(end, frame_type), -249, frame_type)
                bars250 = await sec.load_bars(start, end, frame_type)
                ma60 = signal.moving_average(bars250['close'], 60)
                ma120 = signal.moving_average(bars250['close'], 120)
                ma250 = signal.moving_average(bars250['close'], 250)

                # 上方无均线压制
                if (c0 > ma60[-1]) and (c0 > ma120[-1]) and (c0 > ma250[-1]):
                    logger.info("%s, %s, %s, %s, %s, %s", sec, round(a, 4), round(b, 4),
                                round(vx, 1), round(c0 / c1 - 1, 3), round(y, 3))
                    await self.enter_stock_pool(code, frame_type, end, {
                        "a":   a,
                        "b":   b,
                        "err": err,
                        "y":   y,
                        "vx":  vx
                    })

    async def visualize(self, code: Union[str, List[str]],
                        frame: Union[str, Frame],
                        frame_type: Union[str, FrameType]):
        """
        将code列表中的股票的动量特征图象化
        Args:
            code:
            frame:
            frame_type:

        Returns:

        """
        import matplotlib.pyplot as plt

        if isinstance(code, str):
            code = [code]

        col = 4
        row = len(code) // col + 1
        plt.figure(figsize=(5 * row * col, 7))
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        fit_win = 7
        colors = {
            "5":  '#808080',
            "10": '#00cc80',
            "20": '#00ccff'
        }

        frame = arrow.get(frame)
        frame_type = FrameType(frame_type)
        for i, code in enumerate(code):
            _code = code.split(".")[0]
            start = tf.shift(frame, -25, frame_type)
            bars = await Security(code).load_bars(start, frame, frame_type)

            plt.subplot(len(code) // col + 1, col, i + 1)
            y_lim = 0
            text = ""
            for win in [5, 10, 20]:
                ma = signal.moving_average(bars['close'], win)
                _ma = ma[-fit_win:]

                plt.plot(_ma, color=colors[f"{win}"])

                err, (a, b, c), (vx, _) = signal.polyfit(_ma / _ma[0])
                p = np.poly1d((a * _ma[0], b * _ma[0], c * _ma[0]))
                y = p(fit_win + 2) / p(fit_win - 1) - 1

                y_lim = max(y_lim, np.max(_ma))
                if win == 5:
                    text = f"{_code} a:{a:.4f} b:{b:.4f} vx:{vx:.1f} y:{y:.2f}"

                if err < self.baseline("err", frame_type, f"ma{win}"):
                    # 如果拟合在误差范围内，则画出拟合线
                    plt.plot([p(i) for i in range(len(_ma))], "--",
                             color=colors[f"{win}"])
                    plt.plot([p(i) for i in range(len(_ma))], "o",
                             color=colors[f"{win}"])

                    if 0 < vx < fit_win:
                        plt.plot([vx], p(vx), 'x')

            plt.plot(0, y_lim * 1.035)
            plt.text(0.1, y_lim * 1.02, text, color='r')

    async def evaluate(self, code: str, frame_type: str = '30m',
                       dt: str = None,
                       win=5,
                       flag='long'):
        """
        如果股价从高点下来，或者从低点上来，则发出信号。高点和低点的确定，由于曲线拟合的原因，
        可能产生上一周期未发出信号，这一周期发出信号，但高点或者低点已在几个周期之前。这里的
        策略是，在新的趋势未形成之前，只报一次
        Args:
            code:
            frame_type:
            dt:
            win:
            flag:

        Returns:

        """
        stop = arrow.get(dt, tzinfo=cfg.tz) if dt else arrow.now(tz=cfg.tz)
        frame_type = FrameType(frame_type)

        bars = await self.get_bars(code, win + self.fit_win, frame_type, stop)

        ma = signal.moving_average(bars['close'], win)
        _ma = ma[-self.fit_win:]
        err, (a, b, c), (vx, _) = signal.polyfit(_ma / _ma[0])
        logger.debug("%s, %s, %s, %s, %s", code, err, a, b, vx)
        if err > self.baseline("err", frame_type, f"ma{win}"):
            self.remember(code, frame_type, "trend", "dunno")
            return

        p = np.poly1d((a, b, c))
        y = p(self.fit_win + 2) / p(self.fit_win - 1) - 1

        previous_status = self.recall(code, frame_type, "trend")

        # 如果b > 10 * a * x，则走势主要由b决定。这里x即fit_win序列，我们向后看3周期
        if abs(b) > 10 * (self.fit_win + 3) * abs(a):
            if b > 0 and previous_status != "long" and flag in ['both', "long"]:
                await self.fire_trade_signal('long', code, stop, frame_type, a=a, b=b,
                                             err=err, vx=vx, y=y)
            if b < 0 and previous_status != "short" and flag in ['both', "short"]:
                await self.fire_trade_signal('short', code, stop, frame_type, err=err,
                                             a=a, b=b, y=y)
            return

        t1 = int(vx) < self.fit_win - 1

        t2 = a > self.baseline("a", frame_type, f"ma{win}")
        if t1 and t2 and previous_status != "long" and flag in ["long", "both"]:
            await self.fire_trade_signal('long', code, stop, frame_type, err=err,
                                         a=a, b=b, y=y)
        t2 = a < -self.baseline("a", frame_type, f"ma{win}")
        if t1 and t2 and previous_status != "short" and flag in ["short", "both"]:
            await self.fire_trade_signal('short', code, stop, frame_type, err=err,
                                         a=a, b=b, y=y)

    async def copy(self, code:str, frame_type:Union[str,FrameType],
                   frame: Union[str,Frame],
                   ma_wins=None):
        frame_type = FrameType(frame_type)
        frame = arrow.get(frame, tzinfo=cfg.tz)
        ma_wins = ma_wins or [5, 10, 20]
        fit_win = 7

        stop = frame
        start = tf.shift(stop, -fit_win - max(ma_wins), frame_type)

        sec = Security(code)
        bars = await sec.load_bars(start, stop, frame_type)
        features = []
        for win in ma_wins:
            ma = signal.moving_average(bars['close'], win)
            if len(ma) < fit_win:
                raise ValueError(f"{sec.display_name} doesn't have enough bars for "
                                 f"extracting features")

            err, (a, b, c), (vx, _) = signal.polyfit(ma[-fit_win:]/ma[-fit_win])
            features.append((err, (a,b,c), vx))

        return features


