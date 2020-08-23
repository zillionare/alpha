#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import asyncio
import json
import logging

import arrow
import cfg4py
import numpy as np
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.dal import cache
from omicron.models.securities import Securities
from omicron.models.security import Security
from pyemit import emit

from alpha.core import signal, features
from alpha.plots.baseplot import BasePlot

logger = logging.getLogger(__name__)

cfg = cfg4py.get_instance()


class CrossYear(BasePlot):
    """
    5日均线上穿年线后出现的交易机会
    """

    def __init__(self):
        super().__init__()

    async def start(self, scheduler):
        await super().start()

        scheduler.add_job(self.scan, trigger='cron', hour=2)
        scheduler.add_job(self.scan, trigger='cron', hour=14, minute=15)
        scheduler.add_job(self.buy_long_check, trigger='cron', hour=9, minute=25)
        scheduler.add_job(self.buy_long_check, trigger='cron', minute='*/3')

        logger.info("plot crossyear started")

    async def load_watch_list(self):
        today = arrow.now().date()
        start = tf.date2int(tf.day_shift(today, -5))

        holdings = await cache.sys.smembers("holdings")
        records = await cache.sys.hgetall("plots.crossyear")
        if len(records) == 0:
            await self.scan()

        for code, json_info in records.items():
            if code in holdings:
                continue

            info = json.loads(json_info)
            if info['status'] > 1 or info["fired_at"] < start:
                # obsolete or other status
                continue

            self.stock_pool[code] = info

        return self.stock_pool

    async def buy_long_check(self):
        """
        检查票池里的票是否在集合竞价时触发买入条件
        Returns:

        """

        now = arrow.now()
        if not tf.is_trade_day(now):
            return

        minutes = now.datetime.hour * 60 + now.datetime.minute
        if (minutes < 535) or (690 < minutes < 780) or (minutes > 900):
            return

        pool = []
        for code in self.stock_pool.keys():
            pool.append(asyncio.create_task(self._buy_long_check(code, now)))

        await asyncio.gather(*pool)

    async def _buy_long_check(self, code: str, end: Frame):
        start = tf.shift(tf.floor(end, FrameType.DAY), -30, FrameType.DAY)

        sec = Security(code)
        logger.info("checking %s", sec)

        bars = await sec.load_bars(start, end, FrameType.DAY)
        close = bars['close']
        c0 = close[-1]

        # 检查接近5日均线，要求5日内强于均线，均线不能向下（或者拐头趋势）
        ma = signal.moving_average(close, 5)
        err, (a, b, c), (vx, _) = signal.polyfit(ma[-7:] / ma[-7])

        t1 = np.all(close[-5:] > ma[-5:])
        t2 = err < 3e-3
        t3 = a > 5e-4 or (abs(a) < 1e-5 and b > 1e-3)
        t4= (c0-ma[-1]/c0 < 5e-3)
        t5 = vx < 6

        logger.debug("%s 5日：%s, (a,b):%s,%s", sec, [t1,t2, t3, t4, t5], a, b)
        if all([t1, t2, t3, t4, t5]):
            logger.info("fired 5日买入:%s", sec)

            await emit.emit("/alpha/signals/long", {
                "plot":  "crossyear",
                "code":  code,
                "frame": str(end),
                "desc":  "回探5日线",
                "coef":  np.round([a, b], 4),
                "vx":    vx,
                "c": c0,
                "ma": ma[-5:]
            })
            return

        # 检查接近20日线买点
        ma = signal.moving_average(close, 20)
        err, (a, b, c), (vx, _) = signal.polyfit(ma[-10:] / ma[-10])

        t1 = err < 3e-3
        t2 = a > 5e-4 or (abs(a) < 1e-5 and b > 5e-3)
        t3 = (c0 - ma[-1]) < 5e-3
        t4 = vx < 9

        logger.debug("%s 20日：%s, (a,b):%s,%s", sec, [t1,t2, t3, t4], a, b)
        if all([t1, t2, t3, t4]):
            logger.info("fired 20日买入:%s", sec)
            await emit.emit("/alpha/signals/long", {
                "plot":  "crossyear",
                "code":  code,
                "frame": str(end),
                "desc":  "回探20日线",
                "coef":  np.round([a, b], 4),
                "vx":    vx,
                "c": c0,
                "ma": ma[-5:]
            })
            return

        # 检查是否存在30分钟买点
        start = tf.shift(tf.floor(end, FrameType.MIN30), -30, FrameType.MIN30)
        bars = await sec.load_bars(start, end, FrameType.MIN30)

        close = bars['close']
        ma = signal.moving_average(close, 5)
        err, (a, b, c), (vx, _) = signal.polyfit(ma[-7:] / ma[-7])
        t1 = err < 3e-3
        t2 = a > 5e-4 or (abs(a) < 1e-5 and b > 1e-2)
        t3 = vx < 6

        logger.debug("%s 30分钟：%s, (a,b)", sec, [t1,t2, t3, t4, t5], a, b)
        if all([t1, t2, t3]):
            logger.info("fired 30分钟买入:%s", sec)
            await emit.emit("/alpha/signals/long", {
                "plot":  "crossyear",
                "code":  code,
                "frame": str(end),
                "desc":  "30分钟买点",
                "vx":    vx,
                "c": c0,
                "ma": ma[-5:]
            })

    async def scan(self, end: Frame = None, adv_limit=0.3):
        """
        Args:
            end:
            adv_limit: 不包括在win周期内涨幅超过adv_limit的个股

        Returns:

        """
        win = 20
        secs = Securities()
        end = end or tf.floor(arrow.now(), FrameType.DAY)

        results = []
        holdings = await cache.sys.smembers("holdings")
        for i, code in enumerate(secs.choose(['stock'])):
            try:
                if code in holdings:  # 如果已经持仓，则不跟踪评估
                    continue

                sec = Security(code)
                if sec.code.startswith('688') or sec.display_name.find('ST') != -1:
                    continue

                start = tf.day_shift(end, -270)
                bars = await sec.load_bars(start, end, FrameType.DAY)

                close = bars['close']
                ma5 = signal.moving_average(close, 5)
                ma250 = signal.moving_average(close, 250)

                cross, idx = signal.cross(ma5[-win:], ma250[-win:])
                cross_day = bars[-win + idx]['frame']

                if cross != 1:
                    continue

                ma20 = signal.moving_average(close, 20)
                ma120 = signal.moving_average(close, 120)

                # 如果上方还有月线和ma120线，则不发出信号，比如广州浪奇 2020-7-23,泛海控股2020-8-3
                if close[-1] < ma120[-1] or close[-1] < ma20[-1]:
                    continue

                # 计算20日以来大阳次数。如果不存在大阳线，认为还未到上涨时机，跳过
                grl, ggl = features.count_long_lines(bars[-20:])
                if grl == 0:
                    continue

                #
                # # 计算突破以来净余买量（用阳线量减去阴线量来模拟,十字星不计入）
                # bsc = bars[-10 + idx:]  # bars_since_open: included both side
                # ups = bsc[bsc['close'] > (bsc['open'] * 1.01)]
                # downs = bsc[bsc['open'] > (bsc['close'] * 0.99)]
                # balance = np.sum(ups['volume']) - np.sum(downs['volume'])

                # pc = await sec.price_change(cross_day, tf.day_shift(cross_day, 5),
                #                             FrameType.DAY, return_max=True)

                #
                faf = int(win - idx)  # frames after fired
                adv = await sec.price_change(tf.day_shift(end, -win), end,
                                             FrameType.DAY, False)
                if adv > adv_limit:
                    continue

                logger.info(f"{sec}上穿年线\t{cross_day}\t{faf}")
                await cache.sys.hmset_dict("plots.crossyear", {code:json.dumps({
                    "fired_at":  tf.date2int(end),
                    "cross_day": tf.date2int(cross_day),
                    "faf":       faf,
                    "grl":       grl,
                    "ggl":       ggl,
                    "status":    0  # 0 - generated by plots 1 - disabled manually
                })})
                results.append(
                        [sec.display_name, tf.date2int(end), tf.date2int(cross_day), faf, grl,
                         ggl])
            except Exception as e:
                logger.exception(e)

        logger.info("done crossyear scan.")
        return results

cy = CrossYear()

__all__ = ["cy"]
