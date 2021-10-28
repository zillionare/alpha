import datetime
import unittest

import arrow
import cfg4py
import numpy as np
import omicron
from dateutil.tz import tzfile

from alpha.config import get_config_dir
from alpha.core.features import predict_by_moving_average
from alpha.strategies.simlines import SimLines


class TestSimLines(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cfg4py.init(get_config_dir())
        await omicron.init()

    async def test_find_similar(self):
        sim = SimLines("601615.XSHG", "2021-08-20", 20, [30])
        await sim.init()
        results = await sim.find_similar("2021-08-20")
