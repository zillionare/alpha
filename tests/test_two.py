import unittest

import arrow
import os
import cfg4py
import omicron
from omicron.core.lang import async_run
from omicron.core.timeframe import tf
from omicron.core.types import FrameType

from alpha.config import get_config_dir
from alpha.core import signal
from alpha.core.stocks import stocks
from alpha.plots.hrp import Huierpu
from alpha.plots.two import Two

cfg = cfg4py.get_instance()
class MyTestCase(unittest.TestCase):
    @async_run
    async def setUp(self) -> None:
        os.environ[cfg4py.envar] = 'DEV'
        config_dir = get_config_dir()
        cfg4py.init(config_dir)

        await omicron.init()

    def test_slope(self):
        data = [
            ('600139.XSHG', '2020-8-3'),
            ('002651.XSHE', '2020-7-31'),
            ('601216.XSHG', '2020-7-20'),
            ('603069.XSHG', '2020-6-30')
        ]
        two = Two()
        for code, end_dt in data:
            end_dt = arrow.get(end_dt).replace(hour=15)
            bars = stocks.get_bars(code, 14, '1d', end_dt)
            ma = signal.moving_average(bars['close'], 10)
            _slope, err = two.slope(ma)
            print(f"{code}\t{err:.3f}\t{_slope:.2f}")

    @async_run
    async def test_two_fire_long(self):
        end = arrow.get('2020-8-4 10:00:00')
        plot = Two()
        await plot.fire_long(end)

    @async_run
    async def test_two_scan(self):
        end = arrow.get('2020-08-04 10:00:00')
        start = arrow.get('2020-6-4 10:00:00')
        plot = Two()
        await plot.scan(start, end, plot.fire_long, FrameType.MIN30)

    @async_run
    async def test_hrp_scan(self):
        start = arrow.get('2020-7-30')
        end = arrow.get('2020-8-1')
        plot = Huierpu()
        await plot.scan(start, end, plot.fire_long, FrameType.DAY)

if __name__ == '__main__':
    unittest.main()
