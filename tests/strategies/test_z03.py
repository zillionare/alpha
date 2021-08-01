from alpha.strategies.z03 import Z03
from tests import init_test_env
import unittest
import omicron
import numpy as np


class TestZ03(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        await omicron.init()
        return await super().asyncSetUp()

    async def test_extract_features(self):
        z03 = Z03()

        features = await z03.extract_features(total=10)
        print(features)

    def test_transform(self):
        bars = np.array(
            [(i, i * 10) for i in range(1, 260)],
            dtype=[("close", "<f4"), ("volume", "<f4")],
        )

        bars = bars.view(np.recarray)

        z03 = Z03()
        feat = z03.transform(bars, ts_len=10)
        self.assertEqual(70, len(feat))

        bars = np.array(
            [(i, i * 10) for i in range(1, 265)],
            dtype=[("close", "<f4"), ("volume", "<f4")],
        )

        bars = bars.view(np.recarray)
        feat = z03.transform(bars, ts_len=10, res_win=5)
        self.assertEqual(71, len(feat))
