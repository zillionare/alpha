import logging
import unittest

import arrow
import cfg4py
from omicron.core.lang import async_run
from omicron.core.types import FrameType
from omicron.models.securities import Securities

from alpha.plots.longparallel import LongParallel
from alpha.plots.nine import NinePlot
from tests.base import AbstractTestCase

cfg = cfg4py.get_instance()

logger = logging.getLogger(__name__)


class MyTestCase(AbstractTestCase):

    # @async_run
    # async def test_cross_year(self):
    #     plot = CrossYear()
    #     end = arrow.get('2020-8-3').date()
    #     await plot.scan(end)
    #
    #     for minutes in tf.ticks[FrameType.MIN5]:
    #         h,m = minutes//60, minutes%60
    #         logger.info("time: %s:%s", h, m)
    #         end = arrow.get('2020-8-3', tzinfo=cfg.tz).replace(hour=h, minute=m)
    #         start = arrow.get('2020-8-3',tzinfo=cfg.tz).replace(hour=9, minute=35)
    #         day_start = tf.day_shift(end, -29)
    #         for code in plot.stock_pool["code"]:
    #             sec = Security(code)
    #             bars_15 = await sec.load_bars(start, end, FrameType.MIN5)
    #             c = bars_15['close'][-1]
    #             o = bars_15['open'][0]
    #             h = np.max(bars_15['high'])
    #             l = np.max(bars_15['low'])
    #             v = np.sum(bars_15['volume'])
    #             amount = np.sum(bars_15['amount'])
    #             bars_day = await sec.load_bars(day_start, end.date(), FrameType.DAY)
    #             bars_day['close'][-1] = c
    #             bars_day['open'][-1] = o
    #             bars_day['high'][-1] = h
    #             bars_day['low'][-1] = l
    #             bars_day['volume'][-1] = v
    #             bars_day['amount'][-1]= amount
    #
    #             await plot.on_new_data(sec, bars_day)

    @async_run
    async def test_longparallel(self):
        plot = LongParallel()
        codes = Securities().choose(['stock'])
        for code in codes:  # ['000012.XSHE']:
            dt = arrow.get('2020-6-24')
            await plot.evaluate(code, FrameType.DAY, dt, 15)
        print(plot.results)

    @async_run
    async def test_longparallel_copy(self):
        plot = LongParallel()
        code = '000838.XSHE'
        start = arrow.get('2020-5-26').date()
        stop = arrow.get('2020-7-1').date()
        await plot.copy(code, FrameType.DAY, start, stop)
        # self.assertAlmostEqual(8.15e-3, plot.max_distance, places=5)
        logger.info("%s,%s,%s,%s", code, plot.max_distance, plot.fiterr_ma20,
                    plot.fitslp_ma20)

    @async_run
    async def test_nine(self):
        nine = NinePlot()
        end = arrow.get('2020-8-17')
        await nine.copy('002036.XSHE', FrameType.DAY, end, 5)
        #await nine.scan(5, FrameType.DAY, end=end)
        await nine.scan(arrow.get('2020-8-21').date())


if __name__ == '__main__':
    unittest.main()
