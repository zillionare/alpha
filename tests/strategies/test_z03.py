import unittest

import numpy as np
import omicron

from alpha.strategies.z03 import Z03
from tests import init_test_env


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
        close = np.array([i for i in range(1, 266)])
        volume = np.array([i * 10 for i in range(1, 266)])

        z03 = Z03()
        feat = z03.transform(close, volume, ts_len=10)
        self.assertEqual(70, len(feat))

        close = np.array([i for i in range(1, 265)])
        volume = np.array([i * 10 for i in range(1, 265)])

        feat = z03.transform(close, volume, ts_len=10, res_win=5)
        self.assertEqual(71, len(feat))

    def test_pred(self):
        s = Z03("/tmp/z03_model.pkl")

        _, (X_test, y_test) = s.load_train_data("/tmp/z03_train.pkl")

        for i, x in enumerate(X_test):
            pred = s.predict(x)
            print(pred, y_test[i])
