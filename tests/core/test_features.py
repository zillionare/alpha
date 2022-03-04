import datetime
import itertools
import os
import pickle
import unittest

import numpy as np

from alpha.core.features import *
from tests import data_dir, load_bars_from_file


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

        # ts[11:13]: 1.261, 1.218, 1.253
        ts = [f(i) for i in range(11)]

        preds, pmae = predict_by_moving_average(ts, 5)
        self.assertAlmostEqual(1.261, preds[0])
        self.assertTrue(preds[0] / f(11) - 1 < 0.007)
        self.assertTrue(pmae < 0.001)

        preds, pmae = predict_by_moving_average(ts, 5, 3)
        np.testing.assert_array_almost_equal([1.261, 1.308, 1.359], preds, 3)
        self.assertTrue(pmae < 0.001)

        ts = [f(i) for i in range(19)]
        # to predict ts[19], origin is 1.741, should predict as 1.773
        preds, pmae = predict_by_moving_average(ts, 10)
        self.assertAlmostEqual(1.773, preds[0], 2)
        self.assertTrue(pmae < 0.001)

        ts = [f(i) for i in range(39)]
        # preds, pmae = predict_by_moving_average(ts, 10, 5)
        # np.testing.assert_array_almost_equal((4.27, 4.44, 4.6, 4.77, 4.95), preds, 2)

        preds, pmae = predict_by_moving_average(ts, 20, 5)
        np.testing.assert_array_almost_equal((4.21, 4.37, 4.54, 4.7, 4.87), preds, 2)

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

    def test_peaks_and_valleys(self):
        ts = np.array(
            [
                3589.86,
                3586.2,
                3587.35,
                3587.0,
                3590.6,
                3593.53,
                3602.47,
                3603.62,
                3595.87,
                3582.53,
                3587.56,
                3594.78,
                3596.02,
                3591.88,
                3586.08,
                3598.18,
                3593.5,
                3587.3,
                3584.57,
                3582.6,
                3588.16,
                3586.72,
                3592.24,
                3596.06,
                3593.38,
                3597.39,
                3603.25,
                3609.86,
                3610.58,
                3618.01,
                3615.55,
                3612.88,
                3603.94,
                3597.5,
                3599.46,
                3597.64,
                3575.2,
                3565.64,
                3559.96,
                3564.71,
                3561.38,
                3559.98,
                3555.47,
                3562.31,
                3535.2,
                3528.66,
                3532.11,
                3529.0,
                3524.13,
                3523.6,
                3520.83,
                3518.42,
                3505.15,
                3515.32,
                3525.08,
                3523.94,
                3531.4,
                3540.49,
                3544.02,
                3547.34,
                3540.21,
                3539.08,
                3549.08,
                3549.93,
                3555.34,
                3545.89,
                3546.1,
                3544.48,
                3554.43,
                3548.42,
                3537.14,
                3522.33,
                3477.68,
                3482.24,
                3499.03,
                3505.63,
                3497.15,
                3501.23,
                3498.22,
                3492.46,
                3484.18,
                3482.13,
                3502.18,
                3498.54,
                3515.66,
                3514.5,
                3523.32,
                3521.07,
                3526.13,
                3520.53,
                3524.83,
                3526.87,
                3518.77,
                3522.16,
                3514.98,
                3518.57,
                3506.63,
                3506.4,
                3501.08,
                3491.57,
            ],
            dtype=np.float32,
        )

        peaks, valleys = peaks_and_valleys(ts)
        self.assertListEqual([8, 15, 31, 43, 68, 78, 91], peaks.tolist())
        self.assertListEqual([13, 21, 53, 67, 81], valleys.tolist())

    def test_double_bottom(self):
        bars = np.array(
            [
                (
                    datetime.date(2021, 10, 12),
                    3581.3,
                    3583.64,
                    3515.14,
                    3546.94,
                    4.05393748e10,
                    4.61983492e11,
                    1.0,
                ),
                (
                    datetime.date(2021, 10, 13),
                    3543.49,
                    3569.13,
                    3515.65,
                    3561.76,
                    3.25050667e10,
                    4.05020009e11,
                    1.0,
                ),
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )
        self.assertEqual(1, double_bottom(bars))

    def test_double_top(self):
        bars = np.array(
            [
                (
                    datetime.date(2021, 9, 10),
                    3691.19,
                    3722.87,
                    3681.64,
                    3703.11,
                    6.35200433e10,
                    7.59084638e11,
                    1.0,
                ),
                (
                    datetime.date(2021, 9, 13),
                    3699.25,
                    3716.83,
                    3692.82,
                    3715.37,
                    5.57484019e10,
                    6.96192355e11,
                    1.0,
                ),
                (
                    datetime.date(2021, 9, 14),
                    3709.63,
                    3723.85,
                    3655.63,
                    3662.6,
                    5.64952386e10,
                    6.93778088e11,
                    1.0,
                ),
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )
        actual = double_top(bars)
        self.assertEqual(1, actual)

    def test_dark_cloud_cover(self):
        # 中国西电， 2021-10-08
        bars = np.array(
            [
                (
                    datetime.date(2021, 9, 30),
                    6.11,
                    6.5,
                    6.08,
                    6.4,
                    1.47902182e08,
                    9.30570362e08,
                    1.13,
                ),
                (
                    datetime.date(2021, 10, 8),
                    6.6,
                    6.6,
                    6.0599995,
                    6.08,
                    1.17332954e08,
                    7.27529393e08,
                    1.13,
                ),
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )

        actual = dark_cloud_cover(bars)
        self.assertTrue(actual)

        bars = np.array(
            [
                (
                    datetime.date(2021, 9, 24),
                    6.69,
                    6.82,
                    6.36,
                    6.41,
                    1.30129346e08,
                    8.52839497e08,
                    1.13,
                ),
                (
                    datetime.date(2021, 9, 27),
                    6.83,
                    6.8400006,
                    6.01,
                    6.13,
                    1.91771414e08,
                    1.21969780e09,
                    1.13,
                ),
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )

        self.assertTrue(dark_cloud_cover(bars))

    def test_hammer(self):
        bars = np.array(
            [
                (
                    datetime.date(2021, 9, 24),
                    45.75,
                    47.02,
                    43.7,
                    47.02,
                    4448616.0,
                    2.02563918e08,
                    1.04,
                )
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )

        flag, sl = hammer(bars)
        self.assertListEqual([0], flag.tolist())

    def test_inverted_hammer(self):
        bars = np.array(
            [
                (
                    datetime.date(2021, 9, 16),
                    17.89,
                    19.76,
                    17.76,
                    18.73,
                    6314343.0,
                    1.19839305e08,
                    1.01,
                )
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )

        flag, sl = inverted_hammer(bars)
        self.assertListEqual([0], flag.tolist())

    def test_reversal_features(self):
        bars = load_bars_from_file("sh.30m.20111105.200")

        features, columns = reversal_features("000001.XSHG", bars, FrameType.MIN30)
        print(dict(zip(features, columns)))

    def test_peaks_and_valleys(self):
        bars = load_bars_from_file("sh.30m.20111105.200")

        close = bars["close"]
        peaks, valleys = peaks_and_valleys(close[60:])

        self.assertListEqual([4, 15, 31], peaks)
        self.assertListEqual([7, 12, 21], valleys)

    def test_shadow_features(self):
        bars = np.array(
            (9.2, 9.4, 9.0, 9.3),
            dtype=[("open", "<f4"), ("high", "<f4"), ("low", "<f4"), ("close", "<f4")],
        )
        up, down, body = shadow_features(bars)
        self.assertAlmostEqual(up, 0.01, 2)
        self.assertAlmostEqual(down, 0.02, 2)
        self.assertAlmostEqual(body, 0.01, 2)

    def test_long_parallel(self):
        close = np.array(
            [
                3.07,
                3.09,
                3.21,
                3.34,
                3.38,
                3.7,
                3.77,
                3.62,
                3.64,
                3.5,
                3.46,
                3.55,
                3.77,
                3.76,
                3.87,
                4.32,
                4.1,
                4.11,
                3.84,
                3.76,
                3.94,
                4.08,
                4.18,
                4.03,
                4.18,
                4.28,
                4.24,
                4.08,
                4.12,
                4.16,
                4.08,
                4.18,
                4.04,
                3.78,
                3.83,
                3.76,
                3.8,
                3.94,
                3.89,
                3.9,
                3.8,
                3.86,
                4.04,
                3.98,
                4.1,
                4.17,
                4.07,
                4.07,
                4.05,
                4.13,
                4.07,
                4.3,
                4.42,
                4.22,
                4.41,
                4.44,
                4.35,
                4.24,
                4.65,
                4.57,
                4.35,
                4.6,
                4.48,
                4.61,
                4.67,
                4.55,
                4.25,
                4.09,
                4.1,
                4.06,
                4.36,
                4.29,
                4.31,
                4.16,
                4.06,
                4.14,
                3.91,
                3.86,
                3.78,
                3.93,
                4.38,
                4.3,
                4.2,
                4.56,
                4.4,
                4.71,
                4.75,
                4.7,
                5.05,
                6.06,
            ]
        )

        mas = []
        for w in [5, 10, 20]:
            mas.append(moving_average(close, w)[-10:])
        mas = np.array(mas)

        ll = parallel(mas[0:3, :])
        self.assertEqual(5, ll)

        # 大盘2021-11-05 15：00,假多头
        close = np.array(
            [
                3540.21,
                3539.08,
                3549.08,
                3549.93,
                3555.34,
                3545.89,
                3546.1,
                3544.48,
                3554.43,
                3548.42,
                3537.14,
                3522.33,
                3477.68,
                3482.24,
                3499.03,
                3505.63,
                3497.15,
                3501.23,
                3498.22,
                3492.46,
                3484.18,
                3482.13,
                3502.18,
                3498.54,
                3515.66,
                3514.5,
                3523.32,
                3521.07,
                3526.13,
                3520.53,
                3524.83,
                3526.87,
                3518.77,
                3522.16,
                3514.98,
                3518.57,
                3506.63,
                3506.4,
                3501.08,
                3491.57,
            ]
        )

        mas = []
        for w in [5, 10, 20, 30]:
            mas.append(moving_average(close, w)[-10:])
        mas = np.array(mas)

        ll = parallel(mas[1:, :])
        self.assertEqual(0, ll)

    def test_short_parallel(self):
        close = np.array(
            [
                13.589999,
                13.45,
                13.7,
                13.53,
                13.35,
                13.36,
                13.25,
                12.77,
                12.95,
                14.25,
                14.850001,
                14.24,
                13.9,
                14.0199995,
                14.68,
                14.91,
                16.4,
                16.56,
                16.88,
                15.76,
                16.45,
                16.46,
                17.07,
                17.19,
                16.47,
                16.19,
                15.549999,
                15.17,
                15.2,
                15.950001,
                15.28,
                14.199999,
                13.650001,
                13.78,
                12.42,
                11.9,
                11.84,
                11.71,
                11.84,
                11.630001,
            ]
        )

        mas = []
        for w in [5, 10, 20, 30]:
            mas.append(moving_average(close, w)[-10:])

        sl = parallel(mas)
        self.assertEqual(sl, -2)

    def test_piercing_line(self):
        # 中国西电 2021-8-25
        bars = np.array(
            [
                (
                    datetime.date(2021, 8, 24),
                    5.134588,
                    5.1644974,
                    4.965096,
                    5.0049763,
                    1.01742282e08,
                    5.15075322e08,
                    1.13,
                ),
                (
                    datetime.date(2021, 8, 25),
                    4.94,
                    5.23,
                    4.87,
                    5.22,
                    1.14042512e08,
                    5.79181904e08,
                    1.133392,
                ),
            ],
            dtype=[
                ("frame", "O"),
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )

        self.assertTrue(piercing_line(bars))
