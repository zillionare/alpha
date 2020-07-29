#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import numpy as np
import logging
#import matplotlib.pyplot as plt
from alpha.core.enums import CurveType

logger = logging.getLogger(__name__)

def mean_sqrt_error(y_pred, y_true):
    """
    返回预测序列相对于真值序列的标准差。
    Args:
        y_pred:
        y_true:

    Returns:

    """
    return np.sqrt(sum(np.square(y_pred - y_true) / len(y_pred)))

def polyfit(ts, curve: CurveType = CurveType.PARABOLA, decimals=4):
    """
    对给定的时间序列进行二次曲线或者指数曲线拟合。拟合的曲线种类分别为：

    'line':     y = ax + b
    'parabola': y = ax^2 + bx + c
    'exp':      y = ax^b

    返回的结果为 std_err, curve, coef
    其中std_err归一化为相对于输入序列最小值的百分比。对证券分析而言，可以接受的最大的std_err也许是小于1%，序列越长，标准差越低越好。
    curve为'line', 'parabola'或者'exp(onential)'

    如果拟合出来的二次项系数a小于1e-5，则认为该曲线为直线，返回'line’和系数(b,c),而不是(a,b,c)
    """
    x = np.array(list(range(len(ts))))

    std_err_1, curve_1, coef_1, a, b, c = [None] * 6

    # 1. 先尝试按二次曲线拟合。二次曲线可以拟合到反生反转的行情，如圆弧底、圆弧顶；也可以拟合到上述趋势中的单边走势，即其中一段曲线。
    try:
        z = np.polyfit(x, ts, deg=2)
        a, b, c = round(z[0], decimals), round(z[1], decimals), round(z[2], decimals)
        # polyfit给出的残差是各项残差的平方和，这里返回相对于单项的误差比。对股票行情而言，最大可接受的std_err也许是小于1%
        p1 = np.poly1d((a, b, c))
        y_hat = np.array([a * xi ** 2 + b * xi + c for xi in x])
        std_err_1 = round(np.sqrt(sum(np.square(y_hat / ts - 1)) / len(ts)) * 100, decimals)

        curve_1 = CurveType.PARABOLA
        coef_1 = (a, b, c)

        if curve == CurveType.PARABOLA:
            return std_err_1, curve_1, coef_1
    except Exception as e:
        std_err_1 = 1e9
        logger.warning("ts %s caused calculation error.")
        logger.exception(e)

    std_err = None
    # 2. 尝试按指数曲线进行拟合。指数曲线只能拟合到单边上涨或者下跌的行情，不能拟合出带反转的行情。但是对单边行情，有可能拟合的比抛物线
    # 更好。
    try:
        y = np.log(ts)
        # https://stackoverflow.com/a/3433503/13395693 设置权重可以对small values更友好。
        z = np.polyfit(x, y, deg=1, w=np.sqrt(np.abs(y)))
        a, b = round(z[0], decimals), round(z[1], decimals)

        # 此处需要自行计算std_error，polyfit返回的errors不能使用
        p = np.poly1d((a, b))

        y_hat = np.array([np.exp(a * x) * np.exp(b) for x in range(len(ts))])
        std_err = round(np.sqrt(sum(np.square(y_hat / ts - 1)) / len(ts)) * 100, decimals)

        if curve == CurveType.EXP:
            return std_err, curve, (a, b)
    except Exception as e:
        std_err = 1e9
        logger.warning("ts %s caused calculation error", ts)
        logger.exception(e)

    # 未指定曲线类型，取std_err小的
    if std_err_1 < std_err:
        return std_err_1, curve_1, coef_1
    else:
        return std_err, CurveType.EXP, (a, b)


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
    #fig, axes = plt.subplots(nrows=1, ncols=len(ts), figsize=figsize)
    #fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"

    # for i, _ts in enumerate(ts):
    #     axes[i].plot(_ts)


def momemtem(ts, deg=1):
    _ts = ts / ts[0]

    dts = np.diff(_ts)
    if deg == 2:
        return np.diff(dts)
    else:
        return dts