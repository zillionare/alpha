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

    def test_volume_features(self):
        data_file = os.path.join(data_dir(), "300985.pkl")
        with open(data_file, "rb") as f:
            bars = pickle.load(f)

        vec = volume_features(bars)
        exp = [0, 1, -1, 80, 21, 1, 1, 6.6, 10.4]
        np.testing.assert_array_almost_equal(exp, vec, 1)

    def test_relation_with_prev_high(self):
        data_file = os.path.join(data_dir(), "300985.pkl")
        with open(data_file, "rb") as f:
            bars = pickle.load(f)

        features = relation_with_prev_high(bars["close"], 30)
        np.testing.assert_array_almost_equal([-0.067, -0.07], features, 3)

        # 000155, 2021-08-19
        # fmt: off
        close = [
            11.07    , 11.76    , 11.78    , 12.36    , 12.34    , 12.2     ,
            11.94    , 11.88    , 11.68    , 11.33    , 11.47    , 10.84    ,
            10.4     , 10.96    , 10.5     , 10.56    , 11.42    , 12.03    ,
            12.37    , 12.66    , 12.82    , 12.5     , 12.02    , 12.2     ,
            12.33    , 13.07    , 14.379999, 14.11    , 15.39    , 14.260001,
            14.69    , 14.44    , 15.21    , 14.75    , 15.549999, 16.3     ,
            15.95    , 16.28    , 15.91    , 15.19    , 15.16    , 14.820001,
            15.140001, 15.710001, 14.969999, 14.42    , 14.15    , 14.47    ,
            14.31    , 13.780001, 13.86    , 14.      , 14.03    , 13.28    ,
            12.38    , 12.620001, 12.99    , 13.07    , 13.18    , 13.59    ,
            13.16    , 13.190001, 13.47    , 13.65    , 13.809999, 13.29    ,
            14.079999, 15.15    , 14.359999, 15.140001, 15.19    , 15.93    ,
            17.21    , 18.93    , 17.46    , 17.87    , 18.21    , 17.29    ,
            19.02    , 20.35    , 21.02    , 21.16    , 21.77    , 19.59    ,
            18.75    , 19.58    , 20.89    , 20.8     , 20.      , 22.      ,
            24.2     , 25.410002, 24.03    , 24.93    , 26.17    , 25.57    ,
            27.5     , 24.94    , 23.9     , 24.15
        ]
        # fmt: on
        features = relation_with_prev_high(close, 30)
        np.testing.assert_array_almost_equal([-0.133, -0.12], features, 3)

    def test_weighted_moving_average(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        wma = weighted_moving_average(data, 5)
        print(wma)
        ma = moving_average(data, 5)
        print(ma)
