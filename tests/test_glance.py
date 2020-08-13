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
from alpha.core.features import count_buy_limit_event

cfg = cfg4py.get_instance()
class MyTestCase(unittest.TestCase):
    @async_run
    async def setUp(self) -> None:
        os.environ[cfg4py.envar] = 'DEV'

        config_dir = get_config_dir()
        cfg4py.init(config_dir)

        await omicron.init()

    def test_something(self):
        self.assertEqual(True, False)

    @async_run
    async def test_buy_limit_events(self):
        end = arrow.get('2020-8-7').date()
        start = tf.day_shift(end, -9)
        sec = Security('603390.XSHG')
        bars = await sec.load_bars(start, end, FrameType.DAY)
        count, indices = count_buy_limit_event(sec, bars)
        self.assertEqual(count, 1)
        self.assertEqual(arrow.get('2020-7-28').date(), bars['frame'][indices[0]])

        sec = Security('000070.XSHE')
        start = tf.day_shift(end, -29)
        bars = await sec.load_bars(start, end, FrameType.DAY)
        count_buy_limit_event(sec, bars)

if __name__ == '__main__':
    unittest.main()
