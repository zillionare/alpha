from unittest import IsolatedAsyncioTestCase
from tests import init_test_env
from alpha.core.rsi_stats import RsiStats
from omicron.core.types import FrameType
import omicron


class TestRSIStats(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        await omicron.init()

    async def test_calc(self):
        stats = RsiStats(FrameType.DAY)
        await stats.calc()
