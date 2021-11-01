import unittest
import cfg4py
from alpha.config import get_config_dir
import omicron
import arrow
import datetime
import numpy as np
from alpha.strategies.z51 import Z51


class TestZ01(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        cfg4py.init(get_config_dir())
        await omicron.init()
        return await super().asyncSetUp()

    async def test_backtest(self):
        z51 = Z51(holding_days=10, msr=0.5, bcr=0.5)
        summary = await z51.backtest("2021-6-21", "2021-10-15", "603663.XSHG")
        print(summary)

        self.assertEqual(1.0, summary["win_rate"])
