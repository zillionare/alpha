import itertools
import os
import pickle
import unittest

import numpy as np

from alpha.core.features import (
    fillna,
    ma_permutation,
    moving_average,
    pos_encode_v2,
    predict_by_moving_average,
    relation_with_prev_high,
    reverse_moving_average,
    transform_y_by_change_pct,
    volume_features,
    weighted_moving_average,
)
from tests import data_dir


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

    def test_pos_encode_v2(self):
        ts = np.arange(30)
        mas = np.array([moving_average(ts, n)[-1] for n in (5, 10, 20, 30)])

        argpos = np.argsort(mas)
        codec = pos_encode_v2(4, argpos)
        self.assertEqual(1, codec)

        for i in (5, 10, 15, 20):
            ts[i] = 1200 / i

        mas = np.array([moving_average(ts, n)[-1] for n in (5, 10, 20, 30)])
        argpos = np.argsort(mas)
        codec = pos_encode_v2(4, argpos)
        self.assertAlmostEqual(0.0435, codec, places=3)

        for argpos in itertools.permutations([0, 1, 2, 3]):
            print(pos_encode_v2(4, tuple(argpos)))

    def test_transform_by_advance(self):
        ts = [10, 10.1, 10.2, 10.3]
        self.assertEqual(0, transform_y_by_change_pct(ts, (0.95, 1.05)))

        ts = [10, 10.1, 10.2, 10.6]
        self.assertEqual(1, transform_y_by_change_pct(ts, (0.95, 1.05)))

        ts = [10, 10.1, 10.2, 9.5]
        self.assertEqual(-1, transform_y_by_change_pct(ts, (0.95, 1.05)))

    def test_reverse_moving_average(self):
        c = np.arange(10)
        ma = moving_average(c, 3)
        c1 = [reverse_moving_average(ma, i, 3) for i in range(len(ma))]
        exp = [5, 6, 7, 8, 9]
        np.testing.assert_array_almost_equal(exp, c1[3:], 3)

        c = np.arange(15)
        ma = moving_average(c, 5)
        c1 = [reverse_moving_average(ma, i, 5) for i in range(len(ma))]
        exp = np.arange(9, 15).tolist()
        np.testing.assert_array_almost_equal(exp, c1[5:], 3)

    def test_predict_by_moving_average(self):
        def f(i):
            return 0.002 * i * i + 0.001 * i + 1

        ts = [f(i) for i in range(11)]

        preds, pmae = predict_by_moving_average(ts, 5)
        self.assertAlmostEqual(1.261, preds[0])
        self.assertTrue(preds[0] / f(11) - 1 < 0.007)
        self.assertTrue(pmae < 0.001)

        preds, pmae = predict_by_moving_average(ts, 5, 3)
        np.testing.assert_array_almost_equal([1.261, 1.308, 1.359], preds, 3)
        self.assertTrue(pmae < 0.001)

        ts = [f(i) for i in range(19)]
        preds, pmae = predict_by_moving_average(ts, 10)
        self.assertAlmostEqual(1.774, preds[0], 3)
        self.assertTrue(pmae < 0.001)

    def test_volume_features(self):
        data_file = os.path.join(data_dir(), "300985.pkl")
        with open(data_file, "rb") as f:
            bars = pickle.load(f)

        vec = volume_features(bars)
        exp = [0, 1, -1, 1, 0.482, 0.025, 1, 0.278, 0.42]
        np.testing.assert_array_almost_equal(exp, vec, 3)

    def test_relation_with_prev_high(self):
        data_file = os.path.join(data_dir(), "300985.pkl")
        with open(data_file, "rb") as f:
            bars = pickle.load(f)

        features = relation_with_prev_high(bars["close"], 30)
        np.testing.assert_array_almost_equal([1, -0.07], features[-1], 3)

    def test_weighted_moving_average(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        wma = weighted_moving_average(data, 5)
        print(wma)
        ma = moving_average(data, 5)
        print(ma)
