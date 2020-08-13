import os
import unittest

import arrow
import cfg4py
import logging

import omicron
from omicron.core.lang import async_run
from omicron.core.timeframe import tf
from omicron.core.types import FrameType, bars_dtype
from omicron.models.security import Security
import numpy as np

from alpha.config import get_config_dir
from alpha.plots.crossyear import CrossYear

cfg = cfg4py.get_instance()

logger = logging.getLogger(__name__)
class MyTestCase(unittest.TestCase):
    @async_run
    async def setUp(self) -> None:
        os.environ[cfg4py.envar] = 'DEV'
        config_dir = get_config_dir()
        cfg4py.init(config_dir)

        await omicron.init()

    @async_run
    async def test_cross_year(self):
        plot = CrossYear()
        end = arrow.get('2020-8-3').date()
        await plot.scan(end)

        for minutes in tf.ticks[FrameType.MIN5]:
            h,m = minutes//60, minutes%60
            logger.info("time: %s:%s", h, m)
            end = arrow.get('2020-8-3', tzinfo=cfg.tz).replace(hour=h, minute=m)
            start = arrow.get('2020-8-3',tzinfo=cfg.tz).replace(hour=9, minute=35)
            day_start = tf.day_shift(end, -29)
            for code in plot.stock_pool["code"]:
                sec = Security(code)
                bars_15 = await sec.load_bars(start, end, FrameType.MIN5)
                c = bars_15['close'][-1]
                o = bars_15['open'][0]
                h = np.max(bars_15['high'])
                l = np.max(bars_15['low'])
                v = np.sum(bars_15['volume'])
                amount = np.sum(bars_15['amount'])
                bars_day = await sec.load_bars(day_start, end.date(), FrameType.DAY)
                bars_day['close'][-1] = c
                bars_day['open'][-1] = o
                bars_day['high'][-1] = h
                bars_day['low'][-1] = l
                bars_day['volume'][-1] = v
                bars_day['amount'][-1]= amount

                await plot.on_new_data(sec, bars_day)

if __name__ == '__main__':
    unittest.main()
