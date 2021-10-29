import unittest
import cfg4py
from alpha.config import get_config_dir
import omicron
import arrow
import datetime
import numpy as np
from alpha.strategies.z05 import Z05


class TestZ05(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        cfg4py.init(get_config_dir())
        await omicron.init()
        return await super().asyncSetUp()

    async def test_close_order(self):
        order = {
            "order_date": arrow.get("2021-10-18"),
            "buy_price": 15,
            "code": "000001.XSHE",
            "params": {},
        }

        bars = np.array(
            [
                (
                    datetime.datetime(2021, 10, 19, 10, 0),
                    19.15,
                    19.42,
                    19.15,
                    19.3,
                    11864300.0,
                ),
                (
                    datetime.datetime(2021, 10, 19, 10, 30),
                    19.29,
                    19.65,
                    19.26,
                    19.22,
                    11392100.0,
                ),
                (
                    datetime.datetime(2021, 10, 19, 11, 0),
                    19.58,
                    19.68,
                    19.41,
                    19.42,
                    7745200.0,
                ),
                (
                    datetime.datetime(2021, 10, 19, 11, 30),
                    19.42,
                    19.48,
                    19.38,
                    19.4,
                    5264200.0,
                ),
                (
                    datetime.datetime(2021, 10, 19, 13, 30),
                    19.4,
                    19.56,
                    19.4,
                    19.55,
                    9269700.0,
                ),
                (
                    datetime.datetime(2021, 10, 19, 14, 0),
                    19.55,
                    19.63,
                    19.53,
                    19.22,
                    6631800.0,
                ),
                (
                    datetime.datetime(2021, 10, 19, 14, 30),
                    19.56,
                    19.61,
                    19.52,
                    19.58,
                    7195000.0,
                ),
                (
                    datetime.datetime(2021, 10, 19, 15, 0),
                    19.58,
                    19.62,
                    19.53,
                    19.57,
                    8879100.0,
                ),
            ],
            dtype=(
                np.record,
                [
                    ("frame", "O"),
                    ("open", "<f8"),
                    ("high", "<f8"),
                    ("low", "<f8"),
                    ("close", "<f8"),
                    ("volume", "<f8"),
                ],
            ),
        )

        # sell at last frame
        strategy = Z05(holding_days=1)
        strategy.close_order(order, bars)

        self.assertEqual(1, len(strategy.trades))
        trade = strategy.trades[0]
        self.assertAlmostEqual(trade["gains"], 0.305, 3)
        self.assertEqual(bars[-1]["frame"], trade["sell_at"])
        self.assertEqual("expired", trade["close_type"])

        # sell on stop loss, first 19.22
        order["buy_price"] = 20.3
        strategy.close_order(order, bars)
        self.assertEqual(2, len(strategy.trades))
        trade = strategy.trades[1]

        self.assertAlmostEqual(trade["gains"], -0.053, 3)
        self.assertEqual(bars[1]["frame"], trade["sell_at"])
        self.assertEqual("stop_loss", trade["close_type"])

        # 触发rsi卖出信号
        bars[6]["close"] = 30  # this will cause rsi = 96.28
        strategy.close_order(order, bars)
        self.assertEqual(3, len(strategy.trades))
        trade = strategy.trades[2]

        self.assertEqual("rsi", trade["close_type"])
        self.assertAlmostEqual(trade["gains"], 0.478, 3)
        self.assertEqual(bars[6]["frame"], trade["sell_at"])

    async def test_try_open_position(self):
        bars = np.array(
            [
                (datetime.date(2021, 9, 28), 105.97, 108.74, 103.9, 107.15, 617897.0),
                (datetime.date(2021, 9, 29), 106.05, 108.0, 103.95, 103.95, 615200.0),
                (datetime.date(2021, 9, 30), 104.0, 105.68, 104.0, 104.57, 346900.0),
                (datetime.date(2021, 10, 8), 105.88, 106.73, 104.08, 104.18, 370536.0),
                (datetime.date(2021, 10, 11), 105.0, 105.55, 103.11, 104.39, 381006.0),
                (datetime.date(2021, 10, 12), 104.36, 104.36, 101.4, 102.07, 580286.0),
                (datetime.date(2021, 10, 13), 102.49, 104.09, 101.81, 104.07, 429580.0),
                (datetime.date(2021, 10, 14), 104.28, 105.0, 102.89, 103.22, 337100.0),
                (datetime.date(2021, 10, 15), 103.44, 103.93, 102.42, 102.64, 336143.0),
                (datetime.date(2021, 10, 18), 101.3, 106.88, 101.3, 105.62, 712173.0),
                (datetime.date(2021, 10, 19), 104.14, 115.98, 103.58, 114.0, 2013752.0),
                (datetime.date(2021, 10, 20), 111.0, 118.11, 109.31, 113.96, 1968892.0),
            ],
            dtype=(
                np.record,
                [
                    ("frame", "O"),
                    ("open", "<f8"),
                    ("high", "<f8"),
                    ("low", "<f8"),
                    ("close", "<f8"),
                    ("volume", "<f8"),
                ],
            ),
        )

        code = "301051.XSHE"
        z05 = Z05(holding_days=1)
        z05.predict(code, bars)
        self.assertEqual(1, len(z05.orders))

    async def test_backtest(self):
        z05 = Z05(holding_days=3, msr=0.85)
        stocks = ["000422.XSHE"]
        summary = await z05.backtest("2021-10-20", "2021-10-20", stocks)
        print(summary)

        self.assertAlmostEqual(0.10, summary["returns"], 2)
        self.assertEqual(1.0, summary["win_rate"])

        stocks = ["002895.XSHE"]
        summary = await z05.backtest("2021-09-15", "2021-09-15", stocks)
        print(summary)

        self.assertAlmostEqual(0.048, summary["returns"], 2)
        self.assertEqual(0.5, summary["win_rate"])
        order = z05.orders[1]
        self.assertAlmostEqual(-0.0512, order["gains"], 2)
