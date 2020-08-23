#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging
from typing import Union

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


def polyfit(ts, deg=2):
    """
    对给定的时间序列进行二次曲线拟合。二次曲线可以拟合到反生反转的行情，如圆弧底、圆弧顶；也可
    以拟合到上述趋势中的单边走势，即其中一段曲线。

    返回的结果为 error, coef, vertex (axis_x, axis_y)

    为方便比较，error取值为标准差除以时间序列ts的均值，即每个拟合项相对于真值均值的误差比例


    """
    x = np.array(list(range(len(ts))))

    try:
        z = np.polyfit(x, ts, deg=deg)

        # polyfit给出的残差是各项残差的平方和，这里返回相对于单项的误差比。对股票行情而言，最大可接受的std_err也许是小于1%
        p = np.poly1d(z)
        ts_hat = np.array([p(xi) for xi in x])
        error = rmse(ts, ts_hat) / np.sqrt(np.mean(np.square(ts)))

        if deg == 2:
            a, b, c = z[0], z[1], z[2]
            axis_x = -b / (2 * a)
            axis_y = (4 * a * c - b * b) / (4 * a)

            return error, z, (axis_x, axis_y)
        elif deg == 1:
            return error, z
    except Exception as e:
        error = 1e9
        logger.warning("ts %s caused calculation error.")
        logger.exception(e)
        return error, (np.nan, np.nan, np.nan), (np.nan, np.nan)


def slope(ts):
    """
    本函数对长期均线的短区间内拟合更有效，长均线在一个短的区间里呈现单
    调上升或者下降趋势，这样的股票更容易操作。
    为使得各时间序列的斜率可以比较，函数内部对其进行了归一化处理。
    Args:
        ts:

    Returns:

    """
    ts = ts / ts[0]
    err, (a, b) = polyfit(ts, deg=1)
    return a, err


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
        error = 1e9
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


def polyfit_inflextion(ts, win=10, err=0.001):
    """
    通过曲线拟合法来寻找时间序列的极值点（局部极大值、极小值）。

    ts为时间序列， win为用来寻找极值的时间序列窗口。
    erro为可接受的拟合误差。

    Returns:
        极值点在时间序列中的索引值
    """
    valleys = []
    peaks = []
    mn, mx = None, None
    for i in range(win, len(ts) - win):
        _err, coef, vert = polyfit(ts[i - win:i])
        if _err > err:
            continue
        a, b, c = coef
        x, y = vert
        if not (8 <= x <= win - 1):
            continue

        index = i - win + 1 + int(x)
        if a > 0:  # 找到最低点
            value = ts[index]
            if mn is None:
                mn = value
                valleys.append(index)
                continue

            if index - valleys[-1] <= 2:
                if value < mn:  # 相邻位置连续给出信号，合并之
                    valleys.pop(-1)
                    valleys.append(index)
                    mn = value
            else:
                valleys.append(index)
        else:  # 找到最高点
            value = ts[index]
            if mx is None:
                mx = value
                peaks.append(index)
                continue

            if index - peaks[-1] <= 2:
                if value > mx:  # 相邻位置连续给出信号，合并之
                    peaks.pop(-1)
                    peaks.append(index)
                    mx = value
                else:
                    peaks.append(index)

    return peaks, valleys


def cross(f, g):
    """
    判断序列f是否与g相交。如果两个序列有且仅有一个交点，则返回1表明f上交g；-1表明f下交g
    returns:
        (flag, index), 其中flag取值为：
        0 无效
        -1 f向下交叉g
        1 f向上交叉g
    """
    indices = np.argwhere(np.diff(np.sign(f - g))).flatten()

    if len(indices) == 0:
        return 0, 0

    # 如果存在一个或者多个交点，取最后一个
    idx = indices[-1]

    if f[idx] < g[idx]:
        return 1, idx
    elif f[idx] > g[idx]:
        return -1, idx
    else:
        return np.sign(g[idx - 1] - f[idx - 1]), idx


def vcross(f, g):
    """
    判断序列f是否与g存在类型v型的相交。即存在两个交点，第一个交点为向下相交，第二个交点为向上
    相交。一般反映为洗盘拉升的特征。
    Args:
        f:
        g:

    Returns:

    """
    indices = np.argwhere(np.diff(np.sign(f - g))).flatten()
    if len(indices) == 2:
        idx0, idx1 = indices
        if f[idx0] > g[idx0] and f[idx1] < g[idx1]:
            return True, (idx0, idx1)

    return False, (None, None)


def is_curve_up(momentum: float, vx: Union[float, int], win: int):
    """
    在一个起始点为1.0（即已标准化）的时间序列中，如果经过signal.polyfit以后，
        1）a > 0, b > 0， 向上开口抛物线，最低点在序列左侧（vx < 0)
        2) a > 0, b < 0, 向上开口抛物线，序列从vx之后开始向上。即如果vx>=win，则还要等待
            win - vx + 1个周期才能到最低点，然后序列开始向上
        3）a < 0, b > 0, 向下开口抛物线，序列从vx之后开始向下。即如果vx>win，则序列还将向上
            运行一段时间（vx-win+1个frame)后再向下
        4） a < 0, b < 0，向下开口抛物线，最高点在序列左侧(vx < 0)

    观察a,b与曲线顶关系：
        def test_abc(a,b):
        p = np.poly1d((a,b,1))
        x = [i for i in range(10)]
        y = [p(i) for i in range(10)]

        plt.plot(x,y)

        err, (a,b,c),(vx,_) = signal.polyfit(y)
        print(np.round([a,b,c,vx],4))

    由于a,b,c和vx相互决定，c==1.0，因此只需要a和vx两个变量就可以决定曲线未来走向。
    Args:
        momentum: 即二次曲线的系数a
        vx:
        win:

    Returns:

    """
    return (momentum > 0 and vx < win - 1) or (momentum < 0 and vx > win)

def curve_type(err:float, a:float, b:float, vx:Union[float, int], win:int):
    if err > 3e-4:
        return CurveType.UNKNOWN
    if abs(a) < 1e-6:
        assert abs(vx) > 100 * win
        return CurveType.LINE_UP if b > 0 else CurveType.LINE_DOWN

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths
