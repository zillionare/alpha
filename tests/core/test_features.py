import unittest
import numpy as np
from alpha.core.features import fillna, ma_permutation, transform_by_advance


class TestFeatures(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_fillna(self):
        arr = np.arange(10) / 3.0
        arr[0:2] = np.nan

        actual = fillna(arr)
        exp = arr.copy()
        exp[0:2] = 2 / 3.0

        np.testing.assert_array_almost_equal(exp, actual, 3)

        arr = np.arange(10) / 3.0
        arr[2:5] = np.nan

        actual = fillna(arr)
        exp = arr.copy()
        exp[2:5] = 1 / 3.0
        np.testing.assert_array_almost_equal(exp, actual, 3)

        arr = np.array([np.nan] * 5)
        try:
            fillna(arr)
        except ValueError:
            self.assertTrue(True)

    def test_ma_permutation(self):
        ts = np.arange(30)
        codec = ma_permutation(ts, 10, [5, 10, 20])
        self.assertEqual(10, len(codec))
        self.assertTrue(np.all(np.array(codec) == 1.0))

    def test_transform_by_advance(self):
        ts = [10, 10.1, 10.2, 10.3]
        self.assertEqual(0, transform_by_advance(ts, (0.95, 1.05)))

        ts = [10, 10.1, 10.2, 10.6]
        self.assertEqual(1, transform_by_advance(ts, (0.95, 1.05)))

        ts = [10, 10.1, 10.2, 9.5]
        self.assertEqual(-1, transform_by_advance(ts, (0.95, 1.05)))
