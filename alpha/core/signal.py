#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

import numpy as np

# import matplotlib.pyplot as plt
from alpha.core.enums import CurveType

logger = logging.getLogger(__name__)


def rmse(y, y_hat):
    """
    返回预测序列相对于真值序列的标准差。
    Args:
        y:
        y_hat:

    Returns:

    """
    return np.sqrt(np.mean(np.square(y - y_hat)))


def polyfit(ts):
    """
    对给定的时间序列进行二次曲线拟合。二次曲线可以拟合到反生反转的行情，如圆弧底、圆弧顶；也可
    以拟合到上述趋势中的单边走势，即其中一段曲线。

    返回的结果为 error, coef, vertex (axis_x, axis_y)

    为方便比较，error取值为标准差除以时间序列ts的均值，即每个拟合项相对于真值均值的误差比例


    """
    x = np.array(list(range(len(ts))))

    try:
        z = np.polyfit(x, ts, deg=2)
        a, b, c = z[0], z[1], z[2]
        # polyfit给出的残差是各项残差的平方和，这里返回相对于单项的误差比。对股票行情而言，最大可接受的std_err也许是小于1%
        p = np.poly1d((a, b, c))
        ts_hat = np.array([p(xi) for xi in x])
        error = rmse(ts, ts_hat) / np.sqrt(np.mean(np.square(ts)))

        coef = (a, b, c)
        axis_x = -b / (2 * a)
        axis_y = (4 * a * c - b * b) / (4 * a)

        return error, coef, (axis_x, axis_y)
    except Exception as e:
        error = 1e9
        logger.warning("ts %s caused calculation error.")
        logger.exception(e)
        return error, (np.nan, np.nan, np.nan), (np.nan, np.nan)

def exp_fit(ts):
    """"""
    try:
        x = list(range(len(ts)))
        y = np.log(ts)
        # https://stackoverflow.com/a/3433503/13395693 设置权重可以对small values更友好。
        z = np.polyfit(x, y, deg=1, w=np.sqrt(np.abs(y)))
        a, b = z[0], z[1]

        # 此处需要自行计算std_error，polyfit返回的errors不能使用
        p = np.poly1d((a, b))

        ts_hat = np.array([np.exp(a * x) * np.exp(b) for x in range(len(ts))])
        error = rmse(ts, ts_hat) / np.sqrt(np.mean(np.square(ts)))

        return error, (a, b)
    except Exception as e:
        error= 1e9
        logger.warning("ts %s caused calculation error", ts)
        logger.exception(e)
        return error, (None, None)

def moving_average(ts, win):
    return np.convolve(ts, np.ones(win), 'valid') / win


def pma(bars):
    return bars['money'] / bars['volume']


def predict_moving_average(ts, win: int, n: int = 1):
    """
    预测次n日的均线位置。如果均线能拟合到某种曲线，则按曲线进行预测，否则，在假定股价不变的前提下来预测次n日均线位置。
    """
    # 1. 曲线拟合法
    curve_len = 5
    ma = moving_average(ts, win)[-curve_len:]
    deg, coef, error = polyfit(ma)
    prediction = []

    if np.sqrt(error / curve_len) / np.mean(ma) < 0.01:
        p = np.poly1d(coef)
        for i in range(curve_len, curve_len + n):
            prediction.append(p(i))

        return prediction

    # 2. 如果曲线拟合不成功，则使用假定股价不变法
    _ts = np.append(ts, [ts[-1]] * n)
    ma = moving_average(_ts, win)
    return ma[-n:]


def parallel_show(ts, figsize=None):
    """形态比较"""
    figsize = figsize or (20, 20 // len(ts))
    # fig, axes = plt.subplots(nrows=1, ncols=len(ts), figsize=figsize)
    # fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"

    # for i, _ts in enumerate(ts):
    #     axes[i].plot(_ts)


def momemtem(ts, deg=1):
    _ts = ts / ts[0]

    dts = np.diff(_ts)
    if deg == 2:
        return np.diff(dts)
    else:
        return dts
