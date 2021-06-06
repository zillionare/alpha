import unittest

import omicron
from alpha.backtesting.backtest import Backtest
from alpha.backtesting.strategy import Strategy
from omicron.core import talib
from omicron.models.security import Security
from tests import init_test_env, load_bars_from_file


class SmaCross(Strategy):
    fast = 10
    slow = 30

    async def init(self):
        self.sma1 = await self.declare_indicator(
            talib.moving_average, self.data.close, self.fast
        )
        self.sma2 = await self.declare_indicator(
            talib.moving_average, self.data.close, self.slow
        )

    async def next(self):
        flag, index = talib.cross(self.sma1, self.sma2)

        # 刚刚发生上穿
        if flag == 1 and index >= len(self.sma1) - 1:
            await self.buy()
        elif flag == -1 and index >= len(self.sma1) - 1:
            await self.close_position()


class TestBacktest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        await omicron.init()
        return await super().asyncSetUp()

    async def test_run(self):
        df = load_bars_from_file('000001.XSHG', '1d')
        sec = Security('000001.XSHG')
        sec.load_bars_from_dataframe(df)
        bt = Backtest(sec, SmaCross)
        await bt.run()
