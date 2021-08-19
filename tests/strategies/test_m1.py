import os
from tests import data_dir
from alpha.strategies.m1 import (
    _predict,
    _train,
    ma_features,
    rsi_features,
    train,
    volume_features,
)
import arrow
import omicron
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.security import Security
from alpha.config import get_config_dir
import unittest
import numpy as np
import pickle

import cfg4py


class TestM1(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        cfg4py.init(get_config_dir())
        await omicron.init()
        return await super().asyncSetUp()

    async def test_ma_features(self):
        close = []
        for item in """
        26.35 26.28 26.38 26.36 26.22 26.2  26.21 26.18 26.37 26.33 26.46 26.42
        26.45 26.33 26.46 26.42 28.16 28.46 28.46 28.   27.71 28.   27.79 27.76
        26.79 26.4  26.9  26.76 26.77 26.77 26.51 26.31 25.76 25.87 25.68 25.66
        25.57 25.48 25.46 25.43 25.65 26.44 26.35 26.38 26.31 26.19 26.23 25.99
        25.44 25.7  25.85 25.97 25.94 26.22 26.57 26.69 26.57 26.71 26.7  26.83
        26.75 26.79 26.69 26.8  26.92 26.8  26.9  26.88 27.47 28.3  28.45 28.22
        27.88 27.7  27.57 27.72 27.96 27.82 27.72 27.78""".split(
            " "
        ):
            try:
                close.append(float(item))
            except ValueError:
                pass

        vec = ma_features(np.array(close))
        self.assertAlmostEqual(vec[0], 1.012, places=3)
        self.assertAlmostEqual(vec[-1], 0.042, places=3)
        self.assertEqual(35, len(vec))

    async def test_volume_features(self):
        flags = [0] * 80
        flags[16] = 1
        flags[24] = 1
        flags[69] = 1

        volume = []
        for item in """
        612100.  189900.  118100.   56100.  192800.   89900.  104300.  233100.
        317100.  404700.  181800.  155000.  126900.   93600.  115800.  258600.
        1815200.  984100.  294200.  368700.  276700.  161500.  162800.  383200.
        1368500.  539500.  250100.  133700.  123200.  139000.  151800.  405100.
        718800.  274700.  126900.  153200.   92700.  208000.   88200.  218000.
        280800.  458100.  177200.   75400.  109500.  116200.  118300.  270700.
        491768.  132300.  116400.  101900.   86000.  122700.  568838.  415000.
        346100.  222600.  245100.  104700.  159600.  133200.  147200.  345300.
        409600.  257900.  168100.  116300.  292900. 2206300.  577000.  799000.
        863800.  564400.  277000.   92100.  214400.  154400.  172700.  288200.
        """.split(
            " "
        ):
            try:
                volume.append(float(item))
            except ValueError:
                pass

        vec = volume_features(np.array(volume), np.array(flags))
        self.assertEqual(6, len(vec))
        np.testing.assert_array_almost_equal(
            [1, 1, 1, 0.92, 0.89, 0.27], vec, decimal=2
        )

    async def test_rsi_features(self):
        """
        300985, 2021-08-04 15:00往前80个点
        """
        close = []
        for item in """
        26.35 26.28 26.38 26.36 26.22 26.2  26.21 26.18 26.37 26.33 26.46 26.42
        26.45 26.33 26.46 26.42 28.16 28.46 28.46 28.   27.71 28.   27.79 27.76
        26.79 26.4  26.9  26.76 26.77 26.77 26.51 26.31 25.76 25.87 25.68 25.66
        25.57 25.48 25.46 25.43 25.65 26.44 26.35 26.38 26.31 26.19 26.23 25.99
        25.44 25.7  25.85 25.97 25.94 26.22 26.57 26.69 26.57 26.71 26.7  26.83
        26.75 26.79 26.69 26.8  26.92 26.8  26.9  26.88 27.47 28.3  28.45 28.22
        27.88 27.7  27.57 27.72 27.96 27.82 27.72 27.78""".split(
            " "
        ):
            try:
                close.append(float(item))
            except ValueError:
                pass

        vec = rsi_features(np.array(close))
        np.testing.assert_array_almost_equal([0.76, 0.22], vec, decimal=2)

    async def test_train_and_predict(self):
        samples = [
            (-1, "300985.XSHE", "20210817 10:00", "高开低走，放量大阴，RSI反转信号已出，滞胀"),
            (1, "300985.XSHE", "20210809 15:00", "放量涨、缩量跌，均线多头，m5,m10,m20收敛，m60向上支撑"),
        ]

        _bars = pickle.load(open(os.path.join(data_dir(), "300985.pkl"), "rb"))

        sample_bars = []
        for (flag, code, end, desc) in samples:
            if isinstance(end, str):
                end = arrow.get(end)

            end_pos = np.max(np.argwhere(_bars["frame"] <= end.datetime))
            bars = _bars[end_pos - 80 : end_pos + 1]
            sample_bars.append((flag, code, end, bars))

        _train(sample_bars, tf.time2int)

        for (flag, code, end, bars) in sample_bars:
            result = _predict(bars)
            self.assertEqual(flag, result[0][-2])
            self.assertAlmostEqual(0, result[0][-1])
