from unittest import IsolatedAsyncioTestCase

import omicron
from coretypes import FrameType

from alpha.core.rsi_stats import RsiStats
from tests import init_test_env


class TestRSIStats(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        await omicron.init()

    async def test_calc(self):
        stats = RsiStats(FrameType.DAY)
        await stats.calc()
