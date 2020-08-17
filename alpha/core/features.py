#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging
from typing import List

import arrow
import matplotlib.pyplot as plt
import numpy as np
from omicron.core.timeframe import tf
from omicron.core.types import FrameType, Frame
from omicron.models.security import Security

from alpha.core import signal

logger = logging.getLogger(__name__)


def count_buy_limit_event(sec: Security, bars: np.array):
    if sec.code.startswith('688'):
        limit = 0.2
    elif sec.display_name.find("ST") != -1:
        limit = 0.05
    else:
        limit = 0.1

    zt = (bars['close'][1:] + 0.01) / bars['close'][:-1] - 1 > limit
    if np.count_nonzero(zt) > 0:
        return np.count_nonzero(zt), (np.argwhere(zt) + 1)[0].flatten()
    else:
        return 0, None


async def summary(code: str, end: Frame, frame: FrameType = FrameType.DAY,
                  win: int = 7):
    sec = Security(code)
    start = tf.shift(end, -(60 + 19), frame)
    bars = await sec.load_bars(start, end, frame)
    end_dt = bars['date'].iat[-1]

    ma5 = np.array(signal.moving_average(bars['close'], 5))
    ma10 = np.array(signal.moving_average(bars['close'], 10))
    ma20 = np.array(signal.moving_average(bars['close'], 20))
    ma60 = np.array(signal.moving_average(bars['close'], 60))

    _ma5, _ma10, _ma20 = ma5[-win:], ma10[-win:], ma20[-win:]

    xlen = 20
    ax_x = [i for i in range(xlen - win, xlen + 1)]  # 回归线的坐标
    # 5日均线及拟合点、预测点
    color = '#b08080'
    lw = 0.5
    err5, coef5, vertex5 = signal.polyfit(_ma5)
    vx, vy = vertex5
    vx = xlen - win + vx
    plt.plot(vx, vy * 0.995, "^", color=color)  # 5日低点

    plt.plot(ma5[-xlen:], color=color, linewidth=lw)  # 均线
    p5 = np.poly1d(coef5)
    y5 = [p5(i) for i in range(win + 1)]

    c0 = bars['close'].iat[-1]
    pred_c = y5[-1] * 5 - bars['close'][-4:].sum()

    plt.gcf().text(0.15, 0.85, f"{code} {end_dt} ^{100 * (pred_c / c0 - 1):.02f}%")
    plt.plot(ax_x, y5, 'o', color=color, mew=0.25, ms=2.5)  # 回归均线

    # 10日均线及回归线
    color = '#00ff80'
    err10, coef10, vertex10 = signal.polyfit(_ma10)
    p10 = np.poly1d(coef10)
    y10 = [p10(i) for i in range(win + 1)]
    plt.plot(ma10[-xlen:], color=color, linewidth=lw)
    plt.plot(ax_x, y10, 'o', color=color, mew=0.25, ms=2.5)

    # 20日均线及回归线
    color = '#00ffff'
    err20, coef20, vertex20 = signal.polyfit(_ma20)
    p20 = np.poly1d(coef20)
    y20 = [p20(i) for i in range(win + 1)]
    plt.plot(ma20[-xlen:], color=color, linewidth=lw)
    plt.plot(ax_x, y20, 'o', color=color, mew=0.25, ms=2.5)

    # 60日均线
    color = "#2222ff"
    plt.plot(ma60[-xlen:], color=color, linewidth=lw)


async def predict_ma(code: str, frame_type: FrameType = FrameType.DAY,
                     end: Frame = None):
    """
    预测ma5、ma10、ma20的下一个数据
    Args:
        code:
        frame_type:
        end:

    Returns:

    """
    sec = Security(code)
    start = tf.shift(end, -29, frame_type)
    bars = await sec.load_bars(start, end, frame_type)

    c0 = bars['close'][-1]

    target = [c0 * (1 + f / 100) for f in range(-3, 3)]

    for c in target:
        close = np.append(bars['close'], c)
        ma5 = signal.moving_average(close, 5)
        ma10 = signal.moving_average(close, 10)
        ma20 = signal.moving_average(close, 20)

        fig = plt.figure()
        axes = plt.subplot(111)
        axes.plot([i for i in range(27)], close[-27:], color='#000000',
                  linewidth=0.5)  # len(ma5) == 27
        axes.plot(ma5)
        axes.plot([i for i in range(5, len(ma10) + 5)], ma10)
        axes.text(26, ma10[-1], f"{ma10[-1]:.2f}")
        axes.plot([i for i in range(15, len(ma20) + 15)], ma20)
        axes.text(26, c, f"{c:.2f}")
        axes.text(0.5, 3450,
                  f"{100 * (c / c0 - 1):.2f}% {c:.2f} {ma5[-1]:.2f} {ma10[-1]:.2f}")


async def plot_ma(code: str, groups=None, end: Frame = None,
                  frame_type: FrameType = FrameType.DAY):
    groups = groups or [5, 10, 20, 60, 120]
    sec = Security(code)
    end = end or tf.floor(arrow.now(), frame_type)
    start = tf.shift(end, -(groups[-1] + 19), frame_type)
    bars = await sec.load_bars(start, end, frame_type)

    for win in groups:
        ma = signal.moving_average(bars['close'], win)
        plt.plot(ma[-20:])


def count_long_lines(bars: np.array):
    """
    计算序列bars中的大阳线和大阴线。实体幅度在7%以上的即为大阳线、大阴线。
    Args:
        bars:

    Returns:

    """
    return np.count_nonzero((bars['close'] - bars['open']) / bars['open'] >= 0.07), \
           np.count_nonzero((bars['open'] - bars['close']) / bars['open'] >= 0.07)


def ma_lines_trend(bars:np.array, ma_wins:List[int]):
    """
    从bars数据中提取均线的走势特征
    Args:
        bars:
        ma_wins:

    Returns:

    """
    features = {}
    for win in ma_wins:
        ma = signal.moving_average(bars['close'], win)
        fit_win = 7 if win == 5 else 10
        err, (a, b, c), (vx, _) = signal.polyfit(ma[-fit_win:] / ma[-fit_win])
        p = np.poly1d((a,b,c))

        # 预测一周后均线涨幅
        war = p(fit_win + 5 - 1)/p(fit_win-1) - 1

        features[f"ma{win}"] = [ma, (err, a, b, vx, fit_win, war)]

    return features
