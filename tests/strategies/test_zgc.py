import datetime
import unittest

import arrow
import cfg4py
import numpy as np
import omicron
from dateutil.tz import tzfile

from alpha.config import get_config_dir
from alpha.core.features import predict_by_moving_average
from alpha.strategies.zgc import Zgc


class TestZGC(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cfg4py.init(get_config_dir())
        await omicron.init()

    async def test_scan(self):
        zgc = Zgc()

        # MIN30没有信号
        df = await zgc.scan(codes=["000700.XSHE"], end=arrow.get("2021-09-08 15:00"))
        self.assertEqual(0, len(df))

        df = await zgc.scan(codes=["000931.XSHE"], end=arrow.get("2021-08-27 11:00"))
        self.assertEqual(0, len(df))

        df = await zgc.scan(codes=["000931.XSHE"], end=arrow.get("2021-08-27 14:00"))
        self.assertEqual(0, len(df))

        df = await zgc.scan(codes=["000931.XSHE"], end=arrow.get("2021-08-27 15:00"))
        self.assertEqual(1, len(df))

        df = await zgc.scan(codes=["000520.XSHE"])
        self.assertEqual(0, len(df))
