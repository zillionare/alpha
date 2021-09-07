import datetime
import unittest

import arrow
import cfg4py
import numpy as np
import omicron
from dateutil.tz import tzfile

from alpha.config import get_config_dir
from alpha.core.features import predict_by_moving_average
from alpha.strategies.zgc import ZGCStrategy


class TestZGC(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cfg4py.init(get_config_dir())
        await omicron.init()

    async def test_scan(self):
        zgc = ZGCStrategy()
        df = await zgc.scan(
            frame_type="1d",
            codes = ["000931.XSHE"],
            end= arrow.get("2021-09-2").date()
        )
        self.assertEqual(1, len(df))
