#!/usr/bin/env python

"""Tests for `alpha` package."""


import unittest

import arrow
import jqdatasdk as jq
import omicron
from omicron.core.lang import async_run
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.security import Security
from pandas import DataFrame

from alpha import app
import numpy as np
from alpha.core.signal import moving_average, polyfit


class TestAlpha(unittest.TestCase):
    """Tests for `alpha` package."""

    def setUp(self):
        """Set up evaluate fixtures, if any."""
        jq.auth('18694978299', '8Bu8tcDpEAHJRn')
        self.secs = jq.get_all_securities()

    def all_stocks(self):
        return self.secs.index

    def get_bars(self, code, count, frame, end_dt):
        if not (code.upper().endswith('.XSHE') or code.upper().endswith('.XSHG')):
            if code.startswith('60'):
                code += '.XSHG'
            else:
                code += '.XSHE'
        fields = ['date', 'open', 'high', 'low', 'close', 'volume', 'money', 'factor']
        return jq.get_bars(code, count, frame, fields, include_now=True, end_dt=end_dt,
                           fq_ref_date=end_dt)

    def name_of(self,code:str):
        return self.secs[self.secs.index == code]['display_name'].iat[0]

    def check_buy_limit(self,c1: float, c: float, display_name: str = None):
        """
        股价是否达到涨停价
        :param display_name:
        :param c1: 前一日收盘价
        :param c: 当日收盘价
        :return:
        """
        if display_name and display_name.find("ST") != -1:
            limit = 0.05
        else:
            limit = 0.1

        if not all([c, c1]):
            return False

        return (c + 0.01) / c1 - 1 > limit
    def tearDown(self):
        """Tear down evaluate fixtures, if any."""

    def test_000_something(self):
        """Test something."""
        bars = jq.get_bars('002082.XSHE', 30,df=False,fq_ref_date='2020-7-29',
                           end_dt='2020-7-30')
        ma5 = moving_average(bars['close'], 5)
        ma10 = moving_average(bars['close'], 10)
        ma20 = moving_average(bars['close'], 20)

        print(ma5[-5:])
        print(bars['date'][-1], len(ma5), len(ma10), len(ma20))
        for i in range(16, 21):
            err, curve, coef = polyfit(ma5[i:i+5])
            a,b,c = coef
            axis_x = -b/(2*a)
            axis_y = (4 *a * c - b*b)/(4*a)

            print(f"{bars['date'][i+9]} err:{err:.4f},coef:{a:.3f},{b:.3f}"
                  f",{c:.3f}, "
                  f"(x,"
                  f"y):{axis_x:.1f}"
                  f",{axis_y:.2f}")

        print("=" * 5 + "ma10")
        for i in range(11, 16):
            err, curve, coef = polyfit(ma10[i:i+5])
            a, b, c = coef
            axis_x = -b / (2 * a)
            axis_y = (4 * a * c - b * b) / (4 * a)

            print(f"{bars[i+14]['date']} err:{err:.4f}, coef:{a:.3f},{b:.3f},{c:.3f}, "
                  f"(x,y):{axis_x:.1f}"
                  f",{axis_y:.2f}")

        print("=" * 5 + "ma20")
        for i in range(1, 6):
            err, curve, coef = polyfit(ma20[i:i+5])
            a, b, c = coef
            axis_x = -b / (2 * a)
            axis_y = (4 * a * c - b * b) / (4 * a)

            print(f"{bars[i+24]['date']} err:{err:.4f}, coef:{a:.3f},{b:.3f},{c:.3f}, "
                  f"(x,y):{axis_x:.1f}"
                  f",{axis_y:.2f}")


    @async_run
    async def test_screen(self):
        import cfg4py
        import os
        from alpha.config import get_config_dir
        os.environ[cfg4py.envar] = 'PRODUCTION'
        cfg4py.init(get_config_dir())

        await omicron.init()
        code = '300023.XSHE'
        sec = Security(code)
        stop = arrow.now().datetime
        start = tf.day_shift(stop, -3)

        bars = await sec.load_bars(start, stop, FrameType.DAY)
        print(bars)

        if np.all(bars['close'] > bars['open']):
            print(sec.display_name, "\t",
                  100 * (bars[-1]['close'] / bars[-2]['close'] - 1))
