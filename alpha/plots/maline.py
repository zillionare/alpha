#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import datetime
import logging
from typing import Union

import arrow
from omicron.models.security import Security

from alpha.core.monitors import mm
from omicron.core.timeframe import tf
from omicron.core.types import FrameType

from alpha.core import signal
from alpha.plots.baseplot import BasePlot

logger = logging.getLogger(__name__)


class MaLine(BasePlot):
    """
    根据均线支撑（反压）发出买入或者卖出信号

    超频三， 2020-8-14，周线,8/20触达周线，买入后上涨20%
    ------------------------------
    ma5, err=0.007,a=-0.0003,b=0.0269,vx=50.6	slp3:0.06
    ma10, err=0.003,a=0.0006,b=0.0081,vx=-6.4	slp3:0.05
    ma20, err=0.003,a=0.0002,b=-0.0017,vx=5.4	slp3:0.00

    """

    # adj_60 = 0.015  # 股价接近60日均线，价差不超过1.5%
    # war5 = 0.1      # 5日均线，后5个交易日累计上涨幅度（week advance ratio)，下同
    # war10 = 0.05
    # war20 = 0.02
    # war60 = 0.01
    fit_win = 7

    def __init__(self):
        super().__init__("均线支撑/压力策略")
        self.fired_times = {}
        self.max_fire_time = 5  # 达到最大信号次数后，监控将被取消

    async def check_job_status(self, flag: str, code: str, frame_type: FrameType,
                               win:int):
        key = f"{code}:{frame_type.value}:{flag}"
        count = self.fired_times.get(key, 0)
        count += 1
        self.fired_times[key] = count

        job = mm.find_job(self.name, code, flag, frame_type, win)
        if job is None:
            logger.warning("job not found:%s, %s,%s,%s", self.name, code, flag,
                           frame_type)
        if count >= self.max_fire_time:
            mm.remove(job_name=job.name)
            del self.fired_times[key]
            logger.info("remove job %s due to reach max_fire_time(%s)", job.name, count)

        # disable job for code today, since it's fired
        start = tf.day_shift(arrow.now(), 1)
        start = datetime.datetime(start.year, start.month, start.day, 9, 31)
        mm.reschedule_job(start, job.name)

    async def evaluate(self, code: str, frame_type:Union[str, FrameType]='30m',
                       win:int=5, flag: str='both', slip: float = 0.015):
        """
        测试当前股价是否达到均线(frame_type和win指定）附近
        Args:
            code:
            slip:
            params: frame_type, win, flag

        Returns:

        """
        frame_type = FrameType(frame_type)
        bars = await self.get_bars(code, win, frame_type)
        ma = signal.moving_average(bars['close'], win)

        c0 = bars[-1]['close']
        if abs(c0 / ma[-1] - 1) <= slip:
            await self.fire_trade_signal(flag, code, bars[-1]['frame'], frame_type,
                                         slip=slip, win=win)
            #await self.check_job_status(flag, code, frame_type, win)

    # async def test_ma60_long(self, code: str, bars: np.array):
    #     """
    #     如果股价前期经过大涨，后回调整理，在60日线上获得支撑，为买入信号。
    #
    #     如果近期股价强势，则会表现为短均线成向上抛物线，最低点在60日线上方，但对60线的方向，
    #     不做过多要求，向上，或者平盘，或者向下但近期处于转折点即可。（恒泰艾普，2020-7-29）
    #
    #     如果比较弱势，则会贴着60日线上行，均线粘合。此时短均线再度发散为买点。这种情况要求60
    #     日均线必须向上。
    #
    #     Args:
    #         code:
    #         bars:
    #
    #     Returns:
    #
    #     """
    #     sec = Security(code)
    #     end_dt = bars[-1]['frame']
    #     close = bars['close']
    #     c0 = close[-1]
    #
    #     ma_features = features.ma_lines_trend(bars, [5, 10, 20, 60])
    #     ma5 = ma_features["ma5"][0]
    #     ma10 = ma_features["ma10"][0]
    #     ma20 = ma_features["ma20"][0]
    #     ma60 = ma_features["ma60"][0]
    #
    #     test_group = np.array([c0, ma5[-1], ma10[-1], ma20[-1]])
    #     if not np.all(test_group > ma60[-1]):
    #         logger.debug("%s %s不满足当前价、所有短均线在60线之上的条件", sec, end_dt)
    #         return
    #
    #     # 过去7天里曾出现股价接近60日均线
    #     s60_adj = np.abs(close[:-7] / ma60[:-7] -1)
    #     if not np.any(s60_adj < self.adj_60):
    #         logger.debug("%s %s 7日内股价未曾接近60日线", sec, end_dt)
    #         return
    #
    #     # 测试短期走势是否符合下杀再拉升特征（黄金坑）
    #     err, a5, b5, vx5, fit_win = ma_features["ma5"][1]
    #     t1 = self.is_curve_up(ma_features["ma5"][1], 5)
    #
    #     err, a10, b10, vx10, fit_win = ma_features["ma10"][1]
    #     t2 = err < self.Params.fit10_err and self.is_curve_up(a10, vx10, fit_win, 10)
    #
    #     err, a20, b20, vx20, fit_win = ma_features["ma20"][1]
    #     t3 = err < self.Params.fit20_err and self.is_curve_up(a20, vx, fit_win, 20)
    #
    #     if t1 and t2 and t3:
    #         logger.info("FIRE LONG: %s %s %s 触及60日线，多周期均线向上", sec, end_dt, self.name)
    #         await emit.emit(Events.sig_long, {
    #             "plot":    self.name,
    #             "code":    code,
    #             "fire_on": end_dt,
    #             "desc":    f"{sec.display_name}触发60日均线支撑买入策略。",
    #             "params":  {
    #                 "s60_adj": s60_adj,
    #                 "5":       (a5, b5),
    #                 "10":      (a10, b10),
    #                 "20":      (a20, b20),
    #                 "60":      (a60, b60)
    #             }
    #         })
    #
    #     # 要求60日线方向向上
    #     err, a60, b60, vx, fit_win = ma_features["ma60"][1]
    #     logger.info("%s, %s, %s, %s, %s, %s",
    #                 code, err, self.is_curve_up(a60, b60, vx, fit_win, 60)
    #                 , a60, b60, vx)
    #     if err > self.Params.fit60_err or \
    #             not self.is_curve_up(a60, b60, vx, fit_win, 60):
    #         logger.debug("%s %s 60日均线无法拟合为符合斜率要求的向上直线", sec, end_dt)
    #         return

    def parse_monitor_settings(self, **params):
        """

        Args: params contains these keys:
            code:
            frame_type:
            flag:
            win:
            trigger:

        Returns:

        """
        code = params.get("code")
        trigger = params.get("trigger")
        frame_type = params.get("frame_type")
        win = params.get("win")
        flag = params.get("flag")

        title_keys = ("plot", "code", "frame_type", "flag", "win")
        job_info = {
            "plot": self.name,
            "trigger": trigger,
            "title_keys": title_keys,
            "executor": 'evaluate',
            "executor_params": {
                "code": code,
                "frame_type": frame_type,
                "win": win,
                "flag": flag
            }
        }

        return title_keys, job_info

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

        for k, v in params.items():
            if k == "flag":
                items['flag'] = _flag_map[v]
            elif k == "code":
                items['代码'] = v.split(".")[0]
                items['名称'] = Security(v).display_name
            elif k == "frame_type":
                items['周期'] = _frame_type_map[v]
            elif k == "win":
                items['均线'] = f"MA{v}"

        items['监控计划'] = mm.translate_trigger(trigger)

        return items

