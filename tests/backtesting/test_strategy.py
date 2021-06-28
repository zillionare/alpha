

from alpha.backtesting.forward_array import ForwardArray
from alpha.backtesting.broker import Broker
from alpha.backtesting.strategy import Strategy
import omicron
from tests import init_test_env, load_bars_from_file
import unittest

import logging

logging.basicConfig(level=logging.INFO)

class DummyStrategy(Strategy):
    async def init(self):
        print("inited DummyStrategy")

    async def next(self):
        logging.info("revealed features: %s", len(self.features))

class TestStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        await omicron.init()
        return await super().asyncSetUp()

    async def test_backtest(self):
        data = load_bars_from_file('000001.XSHG', '1d')

        features = ForwardArray(data, name="sh")
        broker = Broker(data=features, cash=10_000, commission = 0.001)
        s = DummyStrategy(broker, features)
        print(await s.backtest())
