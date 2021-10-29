import unittest
from alpha.utils import math_round, buy_limit_price, equal_price
import datetime


class TestUtils(unittest.TestCase):
    def test_round(self):
        self.assertAlmostEqual(math_round(1.514, 2), 1.51)
        self.assertAlmostEqual(math_round(1.515, 2), 1.52)
        self.assertAlmostEqual(math_round(1.516, 2), 1.52)
        # python round will yield 2.67
        self.assertAlmostEqual(math_round(2.675, 2), 2.68)

    def test_buy_limit_price(self):
        close = [6.41, 7.05, 7.76, 9.31, 11.17, 13.40]
        dates = [
            datetime.date(2020, 8, 19),
            datetime.date(2020, 8, 20),
            datetime.date(2020, 8, 21),
            datetime.date(2020, 8, 24),
            datetime.date(2020, 8, 25),
            datetime.date(2020, 8, 26),
        ]

        code = "300313"
        for i in range(len(close) - 1):
            c0, c1 = close[i], close[i + 1]
            date = dates[i]
            self.assertTrue(equal_price(buy_limit_price(code, c0, date), c1))

    def test_equal_price(self):
        self.assertTrue(equal_price(5.01, 5.011))
        self.assertTrue(not equal_price(5.01, 5.02))
        self.assertTrue(equal_price(5.019, 5.0101))
