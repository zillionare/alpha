import os
import unittest

import arrow
import cfg4py
import omicron
from omicron.core.lang import async_run
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.security import Security

from alpha.config import get_config_dir
from alpha.core import signal
from alpha.core.stocks import stocks


class MyTestCase(unittest.TestCase):
    @async_run
    async def setUp(self) -> None:
        os.environ[cfg4py.envar] = 'DEV'
        config_dir = get_config_dir()
        cfg4py.init(config_dir)

        await omicron.init()

    def test_polyfit_inflextion(self):
        bars = stocks.get_bars('000001.XSHG', 100, '30m')

        ma20 = signal.moving_average(bars['close'], 20)
        peaks, valleys = signal.polyfit_inflextion(ma20, 10)

        print(peaks, valleys)

    @async_run
    async def test_cross(self):
        end = arrow.get('2020-7-24').date()
        start = tf.day_shift(end, -270)
        sec = Security('000035.XSHE')
        jlkg = await sec.load_bars(start, end, FrameType.DAY)
        ma5 = signal.moving_average(jlkg['close'], 5)
        ma250 = signal.moving_average(jlkg['close'], 250)

        flag, idx = signal.cross(ma5[-10:], ma250[-10:])
        self.assertEqual(flag, -1)
        self.assertEqual(idx, 8)


if __name__ == '__main__':
    unittest.main()
