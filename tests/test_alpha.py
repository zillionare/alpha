#!/usr/bin/env python

"""Tests for `alpha` package."""


import unittest

import arrow
import jqdatasdk as jq
from pandas import DataFrame

from alpha import alpha
import numpy as np
from alpha.core.signal import moving_average, polyfit


class TestAlpha(unittest.TestCase):
    """Tests for `alpha` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
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
        """Tear down test fixtures, if any."""

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



    def test_screen(self):
        results = self.screen('1d', end_dt='2020-08-05 15:00:00')
        print(results)