#!/usr/bin/env python

"""Tests for `alpha` package."""


import unittest
import jqdatasdk as jq
from alpha import alpha
from alpha.core.signal import moving_average, polyfit


class TestAlpha(unittest.TestCase):
    """Tests for `alpha` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        jq.auth('18694978299', '8Bu8tcDpEAHJRn')

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

