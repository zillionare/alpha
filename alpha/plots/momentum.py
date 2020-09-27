#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import datetime
import json
import logging
from typing import List, Union

import arrow
import cfg4py
import numpy as np
from omicron.core.timeframe import tf
from omicron.core.types import FrameType, Frame
from omicron.dal import cache
from omicron.models.securities import Securities
from omicron.models.security import Security

from alpha.core import signal
from alpha.core.monitors import mm
from alpha.plots.baseplot import BasePlot

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


class Momentum(BasePlot):
    """
    通过动量方法来寻找股票的买入机会，并自动增加买入点监控

    江龙船艇， 2020/8/27
    """

    def __init__(self):
        super().__init__("动能策略")
        self.fit_win = 7
        self.baselines = {
            "up_limit":     0.015,

            "ma5:30m:err":  3e-3,
            "ma5:30m:a":    3e-4,
            "ma5:30m:vx":   (2, 5),
            "ma5:30m:y":    3e-2,

            "ma5:1d:err":   6e-3,
            "ma5:1d:a":     3e-3,
            "ma5:1d:vx":    (3, 5),
            "ma5:1d:y":     5e-2,

            "ma10:1d:err":  3e-3,

            "ma10:30m:err": 3e-4,
            "ma10:30m:a":   1e-4,
            "ma10:30m:b":   1e-3,

            "ma20:1d:err":  3e-3,

            "ma20:30m:err": 3e-4,
            "ma20:30m:a":   13 - 4,
            "ma20:30m:b":   1e-3
        }

    async def scan(self, frame_type: Union[str, FrameType] = FrameType.DAY,
                   end: Frame = None,
                   codes: List[str] = None):
        logger.info("running momentum scan at %s level", frame_type)
        if end is None:
            end = arrow.now(cfg.tz).datetime

        assert type(end) in (datetime.date, datetime.datetime)

        frame_type = FrameType(frame_type)
        ft = frame_type.value
        codes = codes or Securities().choose(['stock'])
        day_bars = {}
        async for code, bars in Security.load_bars_batch(codes, end, 2, FrameType.DAY):
            day_bars[code] = bars

        if len(day_bars) == 0:
            return

        async for code, bars in Security.load_bars_batch(codes, end, 11, frame_type):
            if len(bars) < 11:
                continue

            fired = bars[-1]['frame']
            day_bar = day_bars.get(code)
            if day_bar is None:
                continue

            c1, c0 = day_bars.get(code)[-2:]['close']
            cmin = min(bars['close'])

            # 还处在下跌状态、或者涨太多
            if c0 == cmin or (c0 / c1 - 1) > self.baseline(f"up_limit"):
                continue

            ma5 = signal.moving_average(bars['close'], 5)

            err, (a, b, c), (vx, _) = signal.polyfit(ma5[-7:] / ma5[-7])
            # 无法拟合，或者动能不足
            if err > self.baseline(f"ma5:{ft}:err") or a < self.baseline(f"ma5:{ft}:a"):
                continue

            # 时间周期上应该是信号刚出现，还在窗口期内
            vx_range = self.baseline(f"ma5:{ft}:vx")
            if not vx_range[0] < vx < vx_range[1]:
                continue

            p = np.poly1d((a, b, c))
            y = p(9) / p(6) - 1
            # 如果预测未来三周期ma5上涨幅度不够
            if y < self.baseline(f"ma5:{ft}:y"):
                continue

            sec = Security(code)

            if frame_type == FrameType.DAY:
                start = tf.shift(tf.floor(end, frame_type), -249, frame_type)
                bars250 = await sec.load_bars(start, end, frame_type)
                ma60 = signal.moving_average(bars250['close'], 60)
                ma120 = signal.moving_average(bars250['close'], 120)
                ma250 = signal.moving_average(bars250['close'], 250)

                # 上方无均线压制
                if (c0 > ma60[-1]) and (c0 > ma120[-1]) and (c0 > ma250[-1]):
                    logger.info("%s, %s, %s, %s, %s, %s", sec, round(a, 4), round(b, 4),
                                round(vx, 1), round(c0 / c1 - 1, 3), round(y, 3))
                    await self.enter_stock_pool(code, fired, frame_type,
                                                a=a, b=b, err=err, y=y,
                                                vx=self.fit_win - vx)
            elif frame_type == FrameType.WEEK:
                await self.enter_stock_pool(code, fired, frame_type, a=a, b=b, err=err,
                                            y=y, vx=self.fit_win - vx)
            elif frame_type == FrameType.MIN30:
                await self.fire_trade_signal('long', code, fired, frame_type, a=a, b=b,
                                             err=err, y=y, vx=self.fit_win - vx)

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

                if err < self.baseline(f"ma{win}:{frame_type.value}:err"):
                    # 如果拟合在误差范围内，则画出拟合线
                    plt.plot([p(i) for i in range(len(_ma))], "--",
                             color=colors[f"{win}"])
                    plt.plot([p(i) for i in range(len(_ma))], "o",
                             color=colors[f"{win}"])

                    if 0 < vx < fit_win:
                        plt.plot([vx], p(vx), 'x')

            plt.plot(0, y_lim * 1.035)
            plt.text(0.1, y_lim * 1.02, text, color='r')

    def parse_monitor_settings(self, **params):
        """

        Args:
            **params:

        Returns:

        """
        code = params.get("code")
        trigger = params.get("trigger")
        frame_type = params.get("frame_type")
        win = params.get("win")
        flag = params.get("flag")

        title_keys = ("plot", "code", "frame_type", "flag", "win")
        job_info = {
            "plot":            self.name,
            "trigger":         trigger,
            "title_keys":      title_keys,
            "executor":        'evaluate',
            "executor_params": {
                "code":       code,
                "frame_type": frame_type,
                "win": win,
                "flag":       flag
            }
        }

        return title_keys, job_info

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
            frame_type: frame_type
            dt:
            win:
            flag:

        Returns:

        """
        stop = arrow.get(dt, tzinfo=cfg.tz) if dt else arrow.now(tz=cfg.tz)
        frame_type = FrameType(frame_type)
        ft = frame_type.value

        bars = await self.get_bars(code, win + self.fit_win, frame_type, stop)

        ma = signal.moving_average(bars['close'], win)
        _ma = ma[-self.fit_win:]
        err, (a, b, c), (vx, _) = signal.polyfit(_ma / _ma[0])

        logger.debug("%s, %s, %s, %s, %s", code, err, a, b, vx)
        if err > self.baseline(f"ma{win}:{ft}:err"):
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

        # 判断是否为看多信号
        t2 = a > self.baseline(f"ma{win}:{ft}:a")
        if t1 and t2 and previous_status != "long" and flag in ["long", "both"]:
            await self.fire_trade_signal('long', code, stop, frame_type, err=err,
                                         a=a, b=b, y=y)

        # 判断是否为看空信号
        t2 = a < -self.baseline(f"ma{win}:{ft}:a")
        if t1 and t2 and previous_status != "short" and flag in ["short", "both"]:
            await self.fire_trade_signal('short', code, stop, frame_type, err=err,
                                         a=a, b=b, y=y)

    async def copy(self, code: str, frame_type: Union[str, FrameType],
                   frame: Union[str, Frame],
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

            err, (a, b, c), (vx, _) = signal.polyfit(ma[-fit_win:] / ma[-fit_win])
            features.append((err, (a, b, c), vx))

        return features

    async def list_stock_pool(self, frames: int, frame_types: List[FrameType] = None):
        key = "plots.momentum.pool"

        recs = await cache.sys.hgetall(key)
        items = []
        now = arrow.now()
        for k, v in recs.items():
            frame, code = k.split(":")

            sec = Security(code)
            v = json.loads(v)

            frame_type = FrameType(v.get("frame_type"))

            if frame_type not in frame_types:
                continue

            latest_frame = tf.floor(now, frame_type)
            start = tf.shift(latest_frame, -frames, frame_type)

            fired = tf.int2time(frame) if frame_type in tf.minute_level_frames else \
                tf.int2date(frame)

            if fired < start:
                continue

            items.append({
                "name":  sec.display_name,
                "code":  code,
                "fired": str(fired),
                "frame": frame_type.value,
                "y":     round(v.get("y"), 2),
                "vx":    round(v.get("vx"), 1),
                "a":     round(v.get("a"), 4),
                "b":     round(v.get("b"), 4),
                "err":   round(v.get("err"), 4)
            })

        return {
            "name":    self.display_name,
            "plot":    self.name,
            "items":   items,
            "headers": [
                {
                    "text":  '名称',
                    "value": 'name'
                },
                {
                    "text":  '代码',
                    "value": 'code'
                },
                {
                    "text":  '信号时间',
                    "value": 'fired'
                },
                {
                    "text":  '预测涨幅',
                    "value": 'y'
                },

                {
                    "text":  '动能',
                    "value": 'a'
                },
                {
                    "text":  '势能',
                    "value": 'b'
                },
                {
                    "text":  '周期',
                    "value": 'frame'
                },
                {
                    "text":  '底部距离(周期)',
                    "value": 'vx'
                },
                {
                    "text":  '拟合误差',
                    "value": 'err'
                }
            ]
        }

    def translate_monitor(self, job_name, params: dict, trigger: dict):
        _flag_map = {
            "both":  "双向监控",
            "long":  "做多信号",
            "short": "做空信号"
        }

        _frame_type_map = {
            "1d":   "日线",
            "1w":   "周线",
            "1M":   "月线",
            "30m":  "30分钟线",
            "60m":  "60分钟线",
            "120m": "120分钟线"
        }

        items = {
            "key": job_name
        }

        try:
            for k, v in params.items():
                if k == "flag":
                    items['监控方向'] = _flag_map[v]
                elif k == "code":
                    items['代码'] = v.split(".")[0]
                    items['名称'] = Security(v).display_name
                elif k == "frame_type":
                    items['周期'] = _frame_type_map[v]
                elif k == "win":
                    items['均线'] = f"MA{v}"

            items['监控计划'] = mm.translate_trigger(trigger)

            return items
        except Exception as e:
            logger.exception(e)
            return None
