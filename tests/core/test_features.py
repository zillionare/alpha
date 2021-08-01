import unittest
import numpy as np
from alpha.core.features import fillna


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
