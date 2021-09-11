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
    replace_zero,
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
        self.assertEqual(0, transform_y_by_change_pct(ts, (0.95, 1.05)), 10)

        ts = [10, 10.1, 10.2, 10.6]
        self.assertEqual(1, transform_y_by_change_pct(ts, (0.95, 1.05)), 10)

        ts = [10, 10.1, 10.2, 9.5]
        self.assertEqual(-1, transform_y_by_change_pct(ts, (0.95, 1.05)), 10)

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

        # fmt: off
        ts = [
            44.98, 44.98, 44.98, 44.98, 44.98, 44.98, 44.98, 44.98, 44.98,
            44.98, 44.98, 44.98, 44.98, 44.98, 44.98, 44.98, 44.98, 44.98,
            44.98, 44.98, 44.98, 44.98, 44.98, 44.98, 44.98, 44.98, 44.98,
            44.98, 44.98, 44.98, 44.98, 44.98, 44.98, 44.98, 44.98, 44.98,
            44.98, 44.98, 44.98, 44.98, 38.09, 36.64, 33.85, 33.32, 33.49,
            33.52, 31.88, 31.94, 33.44, 32.25, 31.72, 31.79, 31.85, 32.75,
            32.38, 33.89, 33.42, 31.97, 31.82, 32.09, 32.28, 32.71, 31.83,
            31.32, 30.76, 30.99, 30.3 , 29.79, 28.59, 28.78, 28.  , 29.3 ,
            29.02, 29.36, 28.9 , 28.62, 28.83, 28.94, 28.38, 28.89, 28.23,
            27.92, 28.04, 28.57, 31.79, 30.43, 30.05, 30.18, 29.61, 30.12,
            29.23, 29.48, 27.59, 26.19, 26.19, 26.18, 26.42, 27.76, 26.31,
            25.43, 25.99, 26.69, 26.8 , 28.22, 27.78, 28.25, 29.73, 29.05,
            30.6 , 32.69, 39.23, 47.08, 56.5 , 47.99, 43.7 , 40.66, 39.  ,
            40.14, 39.7 , 40.25
            ]
        # fmt: on
        preds, pmae = predict_by_moving_average(ts, 5, err_threshold=1)
        self.assertAlmostEqual(25.98, preds[0], 2)

    def test_relation_with_prev_high(self):
        # 中材科技， 2021-8-23 创40日新高， 20日不断新高
        close = np.array(
            [
                21.745642,
                21.78491,
                21.559107,
                22.04998,
                21.647465,
                20.675539,
                19.713428,
                19.428724,
                18.85931,
                19.144018,
                19.340368,
                19.33055,
                20.459555,
                20.51846,
                20.704992,
                20.381016,
                20.675539,
                20.498823,
                20.66572,
                21.38,
                22.52,
                22.91,
                22.669998,
                22.709997,
                22.91,
                22.59,
                22.27,
                21.71,
                21.95,
                21.19,
                21.03,
                20.24,
                20.28,
                20.48,
                21.799997,
                21.79,
                23.86,
                24.509998,
                24.43,
                24.91,
                25.22,
                26.17,
                25.5,
                25.410002,
                25.18,
                27.699999,
                28.25,
                29.0,
                28.85,
                29.63,
                29.099998,
                28.83,
                29.35,
                28.99,
                27.6,
                27.62,
                27.8,
                27.6,
                26.7,
                25.75,
                25.75,
                25.469997,
                25.78,
                25.61,
                26.270002,
                24.89,
                27.38,
                27.86,
                28.0,
                27.96,
                27.78,
                28.19,
                27.85,
                28.33,
                27.79,
                27.710001,
                27.680002,
                28.179998,
                28.76,
                29.67,
            ]
        )

        features = relation_with_prev_high(close, 20)
        self.assertEqual(0, features[0])

        features = relation_with_prev_high(close, 40)
        self.assertEqual(29, features[0])

        features = relation_with_prev_high(close[:-5], 20)
        self.assertEqual(-1, features[0])

    def test_weighted_moving_average(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        wma = weighted_moving_average(data, 5)
        print(wma)
        ma = moving_average(data, 5)
        print(ma)

    def test_replace_zero(self):
        arr = np.array([0, 1, 2, 3, 4])
        actual = replace_zero(arr)
        self.assertListEqual([1, 1, 2, 3, 4], actual.tolist())

        arr = np.array([1, 0, 2, 3, 4])
        self.assertListEqual([1, 1, 2, 3, 4], replace_zero(arr).tolist())

        arr = np.array([1, 2, 3, 0, 4])
        self.assertListEqual([1, 2, 3, 3, 4], replace_zero(arr).tolist())

        arr = np.array([1, 2, 3, 4, 0])
        self.assertListEqual([1, 2, 3, 4, 4], replace_zero(arr).tolist())

        arr = np.array([1, 2, 0, 4, 5])
        self.assertListEqual([1, 2, 0.001, 4, 5], replace_zero(arr, 0.001).tolist())
