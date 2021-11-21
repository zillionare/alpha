import datetime
import unittest
import numpy as np
import omicron

from alpha.features.maline import MaLineFeatures
from alpha.notebook import get_bars
from tests import init_test_env


class TestMalineFeatures(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        await omicron.init()

    def test_ma_line_features(self):
        bars = np.array(
            [
                (datetime.date(2021, 8, 31), 9.72, 9.92, 9.64, 9.89, 1874639.0),
                (datetime.date(2021, 9, 1), 9.89, 9.94, 9.66, 9.8, 1397760.0),
                (datetime.date(2021, 9, 2), 9.8, 9.82, 9.65, 9.82, 1171440.0),
                (datetime.date(2021, 9, 3), 9.82, 9.95, 9.81, 9.9, 1478580.0),
                (datetime.date(2021, 9, 6), 9.85, 9.96, 9.84, 9.96, 1057197.0),
                (datetime.date(2021, 9, 7), 9.97, 10.02, 9.9, 9.99, 1188980.0),
                (datetime.date(2021, 9, 8), 10.0, 10.04, 9.94, 10.01, 1149410.0),
                (datetime.date(2021, 9, 9), 10.02, 10.04, 9.89, 9.93, 1409756.0),
                (datetime.date(2021, 9, 10), 9.97, 10.01, 9.83, 9.91, 1273380.0),
                (datetime.date(2021, 9, 13), 9.96, 9.96, 9.71, 9.93, 1245280.0),
                (datetime.date(2021, 9, 14), 9.9, 9.92, 9.68, 9.79, 1496133.0),
                (datetime.date(2021, 9, 15), 9.8, 10.01, 9.72, 9.97, 1415434.0),
                (datetime.date(2021, 9, 16), 9.96, 10.1, 9.91, 9.98, 1671846.0),
                (datetime.date(2021, 9, 17), 9.93, 10.04, 9.73, 9.89, 1302847.0),
                (datetime.date(2021, 9, 22), 10.6, 10.66, 10.11, 10.32, 8451906.0),
                (datetime.date(2021, 9, 23), 10.45, 11.35, 10.35, 11.35, 5399061.0),
                (datetime.date(2021, 9, 24), 11.94, 11.94, 10.85, 11.09, 10812975.0),
                (datetime.date(2021, 9, 27), 11.09, 11.31, 10.5, 10.92, 5284257.0),
                (datetime.date(2021, 9, 28), 10.74, 10.9, 10.46, 10.76, 3049983.0),
                (datetime.date(2021, 9, 29), 10.78, 11.19, 10.54, 11.0, 5061044.0),
                (datetime.date(2021, 9, 30), 11.1, 11.32, 10.81, 11.13, 4225597.0),
                (datetime.date(2021, 10, 8), 11.24, 11.52, 10.75, 11.46, 6005832.0),
                (datetime.date(2021, 10, 11), 11.47, 11.71, 11.25, 11.44, 3396272.0),
                (datetime.date(2021, 10, 12), 11.43, 11.49, 11.05, 11.19, 2563295.0),
                (datetime.date(2021, 10, 13), 11.11, 11.59, 11.06, 11.09, 2538792.0),
                (datetime.date(2021, 10, 14), 11.03, 11.22, 10.95, 11.13, 2292451.0),
                (datetime.date(2021, 10, 15), 11.06, 11.43, 11.0, 11.28, 2659464.0),
                (datetime.date(2021, 10, 18), 11.39, 11.44, 11.09, 11.42, 2378205.0),
                (datetime.date(2021, 10, 19), 11.35, 11.79, 11.3, 11.69, 3386440.0),
                (datetime.date(2021, 10, 20), 11.7, 11.72, 11.35, 11.39, 2042100.0),
                (datetime.date(2021, 10, 21), 11.41, 11.61, 11.18, 11.29, 1881200.0),
                (datetime.date(2021, 10, 22), 11.29, 11.39, 11.03, 11.12, 2060031.0),
                (datetime.date(2021, 10, 25), 11.0, 11.6, 10.95, 11.48, 3052951.0),
                (datetime.date(2021, 10, 26), 11.48, 11.63, 11.19, 11.48, 1923200.0),
                (datetime.date(2021, 10, 27), 11.48, 11.48, 11.11, 11.13, 2037120.0),
                (datetime.date(2021, 10, 28), 11.13, 11.35, 10.83, 11.11, 2629920.0),
                (datetime.date(2021, 10, 29), 11.18, 11.18, 10.68, 10.87, 2266640.0),
                (datetime.date(2021, 11, 1), 10.96, 11.28, 10.62, 11.16, 2052760.0),
                (datetime.date(2021, 11, 2), 11.1, 11.49, 11.08, 11.27, 3560220.0),
            ],
            dtype=(
                [
                    ("frame", "O"),
                    ("open", "<f8"),
                    ("high", "<f8"),
                    ("low", "<f8"),
                    ("close", "<f8"),
                    ("volume", "<f8"),
                ]
            ),
        )
        actual = MaLineFeatures(bars, [5, 10, 20, 30])
        print(actual)

    async def test_ma_line_features_002(self):
        bars = await get_bars("603917.XSHG", 250, "1d", end="2021-11-03")
        actual = MaLineFeatures(bars)
        print(actual)
