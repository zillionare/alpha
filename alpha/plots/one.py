#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

import arrow
import jqdatasdk as jq
from pandas import DataFrame
import numpy as np

from alpha.core.signal import moving_average, polyfit
from alpha.core.stocks import stocks

logger = logging.getLogger(__name__)

class One:
    def screen(self,frame, end_dt=None, adv_lim=25, win=7, a5=0.02, a10=0.001):
        all = []
        fired = []
        if end_dt is None:
            end_dt = arrow.now().datetime

        for i, code in enumerate(stocks.all_stocks()):
            try:
                name = stocks.name_of(code)
                if name.endswith("退"):
                    continue
                if name.find("ST") != -1:
                    continue

                bars = stocks.get_bars(code, 30, frame, end_dt=end_dt)
                if len(bars) == 0:
                    print("get 0 bars", code)
                    continue

                if arrow.get(bars['date'].iat[-1]).date() != arrow.get(end_dt).date():
                    continue

                # 30日涨幅必须小于adv_lim
                if bars['close'].iat[-1] / bars['close'].min() >= 1 + adv_lim / 100:
                    print(f"{code}涨幅大于", adv_lim)
                    continue

                ma5 = np.array(moving_average(bars['close'], 5))
                ma10 = np.array(moving_average(bars['close'], 10))

                err5, coef5, vertex5 = polyfit(ma5[-win:])
                err10, coef10, vertex10 = polyfit(ma10[-win:])

                vx5, _ = vertex5
                vx10, _ = vertex10
                _a5 = coef5[0]
                _a10 = coef10[0]
                all.append([code, _a5, _a10, vx5, vx10, err5, err10])

                # print(code, round_list([err5, vx, pred_up, y5, ma5[-1], y10, ma10[-1]],3))
                # 如果曲线拟合较好，次日能上涨up%以上，10日线也向上，最低点在win/2以内
                t1 = err5 <= 0.003 and err10 <=0.003
                t2 = _a5 > a5 and _a10 > a10
                t3 = (win - 1 > vx5 >= win/2-1) and (vx10 < win/2 - 1)
                if t1 and t2 and t3:
                    c1, c0 = bars['close'].iat[-2], bars['close'].iat[-1]
                    if stocks.check_buy_limit(c1, c0, name):  # 跳过涨停的
                        continue

                    print(f"{stocks.name_of(code)} {code}",[_a5,_a10,vx5,vx10,err5,
                                                           err10])
                    fired.append([code, _a5, _a10, vx5, vx10, err5, err10])
            except Exception as e:
                print(i, e)
                continue
        return DataFrame(data=all,
                         columns=['code', 'a5', 'a10', 'vx5', 'vx10', 'err_5',
                                  'err_10'])