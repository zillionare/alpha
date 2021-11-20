import itertools
import math
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
import ta
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike
from omicron.core.numpy_extensions import find_runs
from omicron.core.talib import cross
from omicron.core.types import FrameType
from scipy.signal import argrelextrema

argpos_permutations = {
    n: list(itertools.permutations(range(n))) for n in (2, 3, 4, 5, 6, 7)
}


def polyfit(ts: np.array, degree: int = 2) -> tuple:
    """fit ts with np.polyfit, return coeff and pmae"""
    coeff = np.polyfit(np.arange(len(ts)), ts, degree)
    pmae = np.abs(np.polyval(coeff, np.arange(len(ts))) - ts).mean() / np.mean(ts)
    return coeff.tolist(), pmae


def reverse_moving_average(ma: ArrayLike, i: int, win: int) -> float:
    """given moving_average, decode the origin value at index i

    if i < win, then return Nan, these values are not in the window thus cannot be recovered

    see also https://stackoverflow.com/questions/52456267/how-to-do-a-reverse-moving-average-in-pandas-rolling-mean-operation-on-pr
    but these func doesn't perfom well with out moving_average
    Example:
        >>> c = np.arange(10)
        >>> ma = moving_average(c, 3)
        >>> c1 = [reverse_moving_average(ma, i, 3) for i in range(len(ma))]
        >>> c1 == [1, 2, 3, 4.9999, 6.000, 7, 8, 9]

    Args:
        ma (np.array): the moving average series
        i (int): the index of origin
        win (int): the window size, which is used to calculate moving average
    """
    return ma[i] * win - ma[i - 1] * win + np.mean(ma[i - win : i])


def predict_by_moving_average(
    ts: ArrayLike, win: int, n_preds: int = 1, err_threshold=1e-2, n: int = None
) -> float:
    """predict the next ith value by fitted moving average

    make sure ts is not too long and not too short

    Args:
        ts (np.array): the time series
        i (int): the index of the value to be predicted, start from 1
        win (int): the window size
        n (int): how many ma sample points used to polyfit the ma line

    Returns:
        tuple: the predicted value and pmae
    """
    ma = moving_average(ts, win)

    # how many ma values used to fit the trendline?
    if n is None:
        n = {5: 7, 10: 10}.get(win, 15)

    if len(ma) < n:
        raise ValueError(f"{len(ma)} < {n}, can't predict")

    coef, pmae = polyfit(ma[-n:], degree=2)
    if pmae > err_threshold:
        return None, None

    # build the trendline with same length as ma
    fitma = np.polyval(coef, np.arange(n - len(ma), n + n_preds))

    preds = [
        reverse_moving_average(fitma[: i + 1], i, win)
        for i in range(len(ma), len(ma) + n_preds)
    ]

    return preds, pmae


def parabolic_flip(ts, rng=7, ma_win=5, calc_ma=True):
    """抛物线转向信号

    与SAR指标不同。本因子利用均线来平滑股价波动，然后将均线拟合成二次曲线，提示顶点（最低点）和趋势。

    如果calc_ma为True，则函数要对`ts`计算moving_average。如果为False,则`ts`为已经计算好的移动均值。使用已经计算好的移动均值可以加快计算速度
    """
    if calc_ma:
        ts = moving_average(ts, ma_win)

    ts_ = ts[-rng:]
    (a, b, c), pmae = polyfit(ts_)

    y_ = np.polyval((a, b, c), np.arange(rng))

    # uncomment this to draw the lines
    # plt.plot(np.arange(len(ts)-rng, len(ts)), y_)
    # plt.plot(ts)

    vx = round(-b / (2 * a), 1)

    flag = 1 if y_[-1] > y_[-2] else -1
    return flag, rng - vx, round(a, 4), round(pmae, 5)


def moving_average(ts: np.array, win: int):
    """计算时间序列ts在win窗口内的移动平均

    Example:

        >>> ts = np.arange(7)
        >>> moving_average(ts, 5)
        >>> array([2.0000, 3.0000, 4.0000])

    """

    return np.convolve(ts, np.ones(win) / win, "valid")


def weighted_moving_average(ts: np.array, win: int) -> np.array:
    """计算加权移动平均

    Args:
        ts (np.array): [description]
        win (int): [description]

    Returns:
        np.array: [description]
    """
    w = [2 * (i + 1) / (win * (win + 1)) for i in range(win)]

    return np.convolve(ts, w, "valid")


def filterna(ts: np.array) -> np.array:
    """从`ts`中去除NaN

    Args:
        ts (np.array): [description]

    Returns:
        np.array: [description]
    """
    return ts[~np.isnan(ts)]


def fillna(ts: np.array):
    """将ts中的NaN替换为其前值

    Args:
        ts (np.array): [description]
    """
    if np.all(np.isnan(ts)):
        raise ValueError("all of ts are NaN")

    if ts[0] is None or math.isnan(ts[0]):
        idx = np.argwhere(~np.isnan(ts))[0]
        ts[0] = ts[idx]

    mask = np.isnan(ts)
    idx = np.where(~mask, np.arange(mask.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return ts[idx]


def replace_zero(ts: np.array, replacement=None) -> np.array:
    """将ts中的0替换为前值, 处理volume数据时常用用到

    如果提供了replacement, 则替换为replacement

    """
    if replacement is not None:
        return np.where(ts == 0, replacement, ts)

    if np.all(ts == 0):
        raise ValueError("all of ts are 0")

    if ts[0] == 0:
        idx = np.argwhere(ts != 0)[0]
        ts[0] = ts[idx]

    mask = ts == 0
    idx = np.where(~mask, np.arange(mask.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return ts[idx]


def pos_encode(stationary: np.array, var: np.array) -> float:
    """
    用于指标排列的位置编码。比如，均线多头排列为ma5 > ma10 > ma20 > ma30 > ma60 > ma120 > ma250, 空头排列为ma5 < ma10 < ma20 < ma30 < ma60 < ma120 < ma250，在这中间还有很多种排列。通过本编码方案，能为每一种排列确定一个惟一的浮点数。

    stationary 一般应已排序。以下显示ma5, ma10, ma20之间各种排序的编码：

    0.0:(5, 10, 20)
    0.2:(5, 20, 10)
    0.23:(10, 5, 20)
    0.53:(10, 20, 5)
    0.9:(20, 5, 10)
    1.0:(20, 10, 5)

    如果 5, 10, 20的排列意味着多头排列的话，则最强的多头排列编码值最小，而最强的空头排列编码值最大，并且编码不重复。
    """
    spectrum = np.sin([1 / (3 ** i) for i in range(len(stationary))])
    maxium = (stationary[::-1] - stationary).dot(spectrum)
    diff = var - stationary
    return diff.dot(spectrum) / maxium


def pos_encode_v2(pos_len: int, argpos: Union[Tuple, List]) -> float:
    """v2 outputs encoded linear increase value in [0, 1]

    Args:
        pos_len (int): [description]
        argpos (np.array): [description]

    Returns:
        float: [description]
    """
    assert 2 <= pos_len <= 7

    permut = argpos_permutations[pos_len]
    return permut.index(tuple(argpos)) / (len(permut) - 1)


def ma_permutation(ts: ArrayLike, n_features: int, ma_groups: List[int]):
    """
    Args:
        bars (list): [description]
    """
    mas = np.array([moving_average(ts, n)[-n_features:] for n in ma_groups])

    stationary = np.arange(len(mas))

    codes = []
    for i in range(n_features):
        pos = np.argsort(mas[:, i])
        codes.append(pos_encode_v2(len(stationary), stationary[pos]))

    return codes


def transform_y_by_change_pct(ts: np.array, watermarks: List[float], ref: Any):
    """根据涨跌幅转换成为标签

    Args:
        ts ([type]): [description]
        watermarks ([type]): 用以分类的threshold，必须是长度为2的升序列表，如(0.95, 1.05)

    """
    c0 = ref

    if c0 == np.NaN or np.all(ts == np.NaN):
        return None

    # 止损优先
    if min(ts) / c0 <= watermarks[0]:
        return -1
    elif max(ts) / c0 >= watermarks[1]:
        return 1
    else:
        return 0


def transform_to_change_pct(ts: np.array) -> np.array:
    ts = fillna(ts)
    return ts[1:] / ts[:-1] - 1


def top_n_argpos(ts: np.array, n: int) -> np.array:
    """get top n (max->min) elements and return argpos which its value ordered in descent

    Example:
        >>> top_n_argpos([4, 3, 9, 8, 5, 2, 1, 0, 6, 7], 2)
        array([2, 3])
    Args:
        ts (np.array): [description]
        n (int): [description]

    Returns:
        np.array: [description]
    """
    return np.argsort(ts)[-n:][::-1]


def relative_strength_index(prices, period=6):
    """使用ta来计算rsi

    需要注意的是，rsi的计算中，递归使用了前一个rsi的值来进行平滑，所以rsi值越到后面越准。因此，为保证长度为m的rsi值的准确性，最好使用 m + period * 3 的长度。本函数已将前period * 3个数值设置为nan，以保证rsi的计算精度。

    Args:
        prices (ArrayLike):
        period (int): default 6
    Returns:
        np.array: rsi with same length as prices, the first period * 3 values are nan
    """
    df = pd.DataFrame({"close": prices})
    assert len(prices) >= period * 3
    rsi = np.round(ta.momentum.rsi(df.close, period).to_numpy(), 2)
    rsi[: period * 3] = np.NaN

    return rsi


def bolling_band(prices, period, num_std_dev=2.0):
    """
    Bollinger bands (BB) are volatility bands placed above and below a moving
    average.
    Volatility is based on the standard deviation, which changes as volatility
    increases and decreases.
    The bands automatically widen when volatility increases and narrow when
    volatility decreases.
    This dynamic nature of Bollinger Bands also means they can be used on
    different securities with the standard settings.
    For signals, Bollinger Bands can be used to identify M-Tops and W-Bottoms
    or to determine the strength of the trend.
    Signals derived from narrowing BandWidth are also important.

    Bollinger BandWidth is an indicator that derives from Bollinger Bands, and
    measures the percentage difference between the upper band and the lower
    band.
    BandWidth decreases as Bollinger Bands narrow and increases as Bollinger
    Bands widen.
    Because Bollinger Bands are based on the standard deviation, falling
    BandWidth reflects decreasing volatility and rising BandWidth reflects
    increasing volatility.

    %B quantifies a security's price relative to the upper and lower Bollinger
    Band. There are six basic relationship levels:
    %B equals 1 when price is at the upper band
    %B equals 0 when price is at the lower band
    %B is above 1 when price is above the upper band
    %B is below 0 when price is below the lower band
    %B is above .50 when price is above the middle band (20-day SMA)
    %B is below .50 when price is below the middle band (20-day SMA)

    They were developed by John Bollinger.
    Bollinger suggests increasing the standard deviation multiplier to 2.1 for
    a 50-period SMA and decreasing the standard deviation multiplier to 1.9 for
    a 10-period SMA.

    http://www.csidata.com/?page_id=797
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_bands
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_band_width
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_band_perce

    Input:
      prices ndarray
      period int > 1 and < len(prices)
      num_std_dev float > 0.0 (optional and defaults to 2.0)

    Output:
      bbs ndarray with upper, middle, lower bands, bandwidth, range and %B

    Test:

    >>> import numpy as np
    >>> prices = np.array([86.16, 89.09, 88.78, 90.32, 89.07, 91.15, 89.44,
    ... 89.18, 86.93, 87.68, 86.96, 89.43, 89.32, 88.72, 87.45, 87.26, 89.50,
    ... 87.90, 89.13, 90.70, 92.90, 92.98, 91.80, 92.66, 92.68, 92.30, 92.77,
    ... 92.54, 92.95, 93.20, 91.07, 89.83, 89.74, 90.40, 90.74, 88.02, 88.09,
    ... 88.84, 90.78, 90.54, 91.39, 90.65])
    >>> period = 20
    >>> print(bb(prices, period))
    [[  9.12919107e+01   8.87085000e+01   8.61250893e+01   5.82449423e-02
        5.16682146e+00   6.75671306e-03]
     [  9.19497209e+01   8.90455000e+01   8.61412791e+01   6.52300429e-02
        5.80844179e+00   5.07661263e-01]
     [  9.26132536e+01   8.92400000e+01   8.58667464e+01   7.55995881e-02
        6.74650724e+00   4.31816571e-01]
     [  9.29344497e+01   8.93910000e+01   8.58475503e+01   7.92797873e-02
        7.08689946e+00   6.31086945e-01]
     [  9.33114122e+01   8.95080000e+01   8.57045878e+01   8.49848539e-02
        7.60682430e+00   4.42420124e-01]
     [  9.37270110e+01   8.96885000e+01   8.56499890e+01   9.00563838e-02
        8.07702198e+00   6.80945403e-01]
     [  9.38972812e+01   8.97460000e+01   8.55947188e+01   9.25117832e-02
        8.30256250e+00   4.63143909e-01]
     [  9.42636418e+01   8.99125000e+01   8.55613582e+01   9.67861377e-02
        8.70228361e+00   4.15826692e-01]
     [  9.45630193e+01   9.00805000e+01   8.55979807e+01   9.95225220e-02
        8.96503854e+00   1.48579313e-01]
     [  9.47851634e+01   9.03815000e+01   8.59778366e+01   9.74461225e-02
        8.80732672e+00   1.93266744e-01]
     [  9.50411874e+01   9.06575000e+01   8.62738126e+01   9.67087637e-02
        8.76737475e+00   7.82660026e-02]
     [  9.49062071e+01   9.08630000e+01   8.68197929e+01   8.89956780e-02
        8.08641429e+00   3.22789193e-01]
     [  9.49015375e+01   9.08830000e+01   8.68644625e+01   8.84332063e-02
        8.03707509e+00   3.05526266e-01]
     [  9.48939343e+01   9.09040000e+01   8.69140657e+01   8.77834713e-02
        7.97986867e+00   2.26311285e-01]
     [  9.48594576e+01   9.09880000e+01   8.71165424e+01   8.50982021e-02
        7.74291521e+00   4.30661576e-02]
     [  9.46722663e+01   9.11525000e+01   8.76327337e+01   7.72280810e-02
        7.03953265e+00  -5.29486389e-02]
     [  9.45543042e+01   9.11905000e+01   8.78266958e+01   7.37753219e-02
        6.72760849e+00   2.48722001e-01]
     [  9.46761721e+01   9.11200000e+01   8.75638279e+01   7.80546993e-02
        7.11234420e+00   4.72660054e-02]
     [  9.45733946e+01   9.11670000e+01   8.77606054e+01   7.47286754e-02
        6.81278915e+00   2.01003516e-01]
     [  9.45322396e+01   9.12495000e+01   8.79667604e+01   7.19508503e-02
        6.56547911e+00   4.16304661e-01]
     [  9.45303313e+01   9.12415000e+01   8.79526687e+01   7.20906879e-02
        6.57766250e+00   7.52141243e-01]
     [  9.43672335e+01   9.11660000e+01   8.79647665e+01   7.02286710e-02
        6.40246702e+00   7.83328285e-01]
     [  9.41460689e+01   9.10495000e+01   8.79529311e+01   6.80194599e-02
        6.19313782e+00   6.21182512e-01]]
    """

    num_prices = len(prices)

    if num_prices < period:
        # show error message
        raise SystemExit("Error: num_prices < period")

    bb_range = num_prices - period + 1

    # 3 bands, bandwidth, range and %B
    bbs = np.zeros((bb_range, 6))

    simple_ma = moving_average(prices, period)

    for idx in range(bb_range):
        std_dev = np.std(prices[idx : idx + period])

        # upper, middle, lower bands, bandwidth, range and %B
        bbs[idx, 0] = simple_ma[idx] + std_dev * num_std_dev
        bbs[idx, 1] = simple_ma[idx]
        bbs[idx, 2] = simple_ma[idx] - std_dev * num_std_dev
        bbs[idx, 3] = (bbs[idx, 0] - bbs[idx, 2]) / bbs[idx, 1]
        bbs[idx, 4] = bbs[idx, 0] - bbs[idx, 2]
        bbs[idx, 5] = (prices[idx] - bbs[idx, 2]) / bbs[idx, 4]

    return bbs


def relation_with_prev_high(close, win=20) -> List:
    """当前bar与前高的关系

    返回二维向量。

    第一维为当前bar到前高的距离。取值范围[-1,win]，当取值为零时，意味着正在不断创新高；为正数时，意味着刚突破前高，数值越大，越远。取-1时，意味着还未突破前高。

    第二维为股价与前高的涨跌幅。

    Args:
        close ([type]): [description]
    Returns:
        relation: [distance, difference]
    """
    vec = []
    close = close[-win:]
    c0 = close[-1]

    if np.isnan(c0):
        raise ValueError("close is nan")

    prev_high_idx = np.argmax(close[:-1])
    prev_high = close[prev_high_idx]

    if c0 > prev_high:
        vec.append((win - 2) - prev_high_idx)
    else:
        vec.append(-1)

    vec.append(c0 / prev_high - 1)

    return vec


def relationship_with_prev_low(close, win=20):
    """当前bar与前低的关系

    第一维为当前bar到前高的距离。取值范围[-1,1]，当取值为零时，意味着正在不断创新高；为正数时，意味着刚突破前高，数值越大，越远。为负，意味着还未突破前高。

    第二维为股价与前高的差值，通过np.tanh进行scale

    Args:
        close ([type]): [description]
        win (int, optional): [description]. Defaults to 20.
    """
    vec = []
    close = close[-win:]
    c0 = close[-1]

    if np.isnan(c0):
        raise ValueError("close is nan")

    prev_low_idx = np.argmin(close[:-1])
    prev_low = close[prev_low_idx]

    if c0 > prev_low:
        vec.append(np.tanh(2 * (win - prev_low_idx - 1) / win))
    else:
        vec.append(-np.tanh(2 * (win - prev_low_idx - 2) / win))

    vec.append(np.tanh(c0 / prev_low - 1))

    return vec


def long_short_features(bars: np.array, flen: int = 10):
    """最近`flen`个bar的阴阳线排列特征

    Args:
        bars (np.array): [description]
    """
    if len(bars) < flen:
        raise ValueError(f"bars length must be larger than {flen}, actual {len(bars)}")

    return np.where(bars["close"] > bars["open"], 1, -1)[-flen:]


def roc_features(close: np.array, flen: int = 10):
    if len(close) < flen + 1:
        raise ValueError(
            f"close length must be larger than {flen} + 1, actual {len(close)}"
        )

    if np.count_nonzero(np.isfinite(close)) != len(close):
        raise ValueError("close should not contains np.nan")

    return (close[1:] / close[:-1] - 1)[-flen:]


def maline_support_ratio(close, ma_win, n) -> Tuple:
    """均线支撑率,即股价在n个周期内，受到均线(`ma_win`)的支撑率

    count(close > ma) / n

    Args:
        close (np.array): 股价
        ma_win (int): 均线周期
        n (int): 在多少个周期内计算支撑率

    Returns:
        Tuple[float,bool]: [0-1], last close > last ma
    """
    ma = moving_average(close, ma_win)[-n:]

    return np.count_nonzero(close[-n:] >= ma) / n, close[-1] >= ma[-1]


def run_length_flip(ts: List[bool]) -> float:
    """计算逻辑值序列`ts`由正到负（或者反之）的翻转，并返回最后翻转的序列长度比。

    如果返回值为正，表明`ts`要么全为正，要么最后的连续序列为正。反之亦然。如果返回值在[-1,1]之间，表明在一个较长的连续序列之后，刚刚发生一个较短的反转序列。返回值的绝对值越大，说明反转之后趋势延续越久。
    """
    flags, pos, length = find_runs(ts)
    if len(flags) == 1:
        sign = -1 if flags[0] else 1
        rlf = sign * length[0]
    else:
        lf, rf = flags[-2:]
        ll, rl = length[-2:]
        sign = 1 if rf else -1
        rlf = sign * rl / ll

    return round(rlf, 2)


def bullish_candlestick_ratio(bars, n) -> float:
    """n个周期里，阳线占比

    Args:
        bars (np.array): 行情数据
        n (int): 周期数

    Returns:
        [float]: [0-1]
    """
    bars_ = bars[-n:]
    return np.count_nonzero(bars_["close"] > bars_["open"]) / n


def is_bullish(bars) -> bool:
    """将`bars`看成一根蜡烛线，判断是否是阳线

    Args:
        bars (np.array): 行情数据

    Returns:
        [bool]: [True, False]
    """
    return bars[-1]["close"] > bars[0]["open"]


def real_body(bars) -> float:
    """将`bars`看成一根蜡烛线，计算实体长度，最终返回以`open`价来计算的比率。

    即：
        last(close)/first(open) - 1

    Args:
        bars (np.array): 行情数据

    Returns:
        [float]: 实体长度
    """
    return bars[-1]["close"] / bars[0]["open"] - 1


def ma_d1(close, win) -> np.array:
    """股价一阶导的移动均值

    Args:
        close (np.array): 股价
        win (int): 均线周期

    Returns:
        [np.array]: 移动平均值
    """
    d1 = close[1:] / close[:-1] - 1

    return moving_average(d1, win)


def ma_d2(close, win) -> np.array:
    """股价二阶导的移动均值

    Args:
        close (np.array): 股价
        win (int): 均线周期

    Returns:
        [np.array]: 移动平均值
    """
    d1 = close[1:] / close[:-1] - 1
    d2 = np.diff(d1)

    return moving_average(d2, win)


def ddv(ts, win) -> Tuple[np.array, np.array]:
    """数组ts移动平均值的一阶和二阶导"""
    ma = moving_average(ts, win)
    d1 = ts[1:] / ts[:-1] - 1
    d2 = np.diff(d1)

    return d1, d2


def max_drawdown(equitity) -> Tuple:
    """计算最大资产回撤

    Args:
        equitity ([type]): [description]

    Returns:
        [Tuple]: mdd, start, send
    """
    i = np.argmax(np.maximum.accumulate(equitity) - equitity)
    j = np.argmax(equitity[:i])

    return (equitity[i] - equitity[j]) / equitity[j], i, j


def rolling(x, win, func):
    results = []
    for subarray in sliding_window_view(x, window_shape=win):
        results.append(func(subarray))

    return np.array(results)


def reversing(close):
    """股价向上/向下反转

    取最近7个周期数据，如果前6周期股价主要在均线以下，最后一周期转到均线之上，称为向上反转；反之则称为向下反转
    Args:
        close ([type]): [description]
    Returns:
        [type]: 0表示无法判断。1表明看涨，-1表明看跌
    """
    ma = moving_average(close, 5)[-7:]
    under_ma = np.count_nonzero(close[-7:-1] <= ma[:-1]) / 6
    above_ma = np.count_nonzero(close[-7:-1] >= ma[:-1]) / 6
    if under_ma > 0.5 and close[-1] > ma[-1]:
        return 1
    elif above_ma and close[-1] < ma[-1]:
        return -1
    return 0


# def williams_r(bars, timeperiod=60) -> np.array:
#     """求Williams %R

#     Args:
#         bars ([type]): [description]
#     Returns:
#         wr at each frame
#     """
#     high = bars["high"].astype("f8")
#     low = bars["low"].astype("f8")
#     close = bars["close"].astype("f8")

#     return talib.WILLR(high, low, close, timeperiod=14)


def down_shadow(bars):
    """求下影线的长度，百分比表示

    Args:
        bars ([type]): [description]
    """
    if len(bars.shape) == 0:
        return min(bars["open"], bars["close"]) / bars["low"] - 1

    base = np.select(bars["open"] > bars["close"], bars["close"], bars["open"])
    return base / bars["low"] - 1


def up_shadow(bars):
    """求上影线长度，百分比表示"""
    if len(bars.shape) == 0:
        return bars["high"] / max(bars["open"], bars["close"]) - 1

    base = np.select(bars["open"] > bars["close"], bars["open"], bars["close"])
    return bars["high"] / base - 1


def double_bottom(bars, win=10, gap=1e-3) -> int:
    """在`win`个周期里，当日最低点是否下探前低且收盘价未创新低？


    Args:
        bars ([type]): [description]
        win (int, optional): [description]. Defaults to 10.

    Returns:
        [int]: 如果为双底，则返回1，否则返回0
    """
    low = bars["low"][-win:-1]
    close = bars["close"][-win:-1]

    ll = np.min(low)
    lc = np.min(close)

    l0 = bars["low"][-1]
    c0 = bars["close"][-1]

    if c0 > lc and abs(l0 / ll - 1) < gap:
        return 1

    return 0


def double_top(bars, win=10, gap=1e-3):
    """在`win`个周期里，当日最高点是否上探前高并且收盘未创新高

    Args:
        bars ([type]): [description]
        win (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: 如果存在双顶，返回1，否则返回0
    """
    high = bars["high"][-win:-1]
    close = bars["close"][-win:-1]

    hh = np.max(high)
    hc = np.max(close)

    h0 = bars["high"][-1]
    c0 = bars["close"][-1]

    if hc > c0 and abs(h0 / hh - 1) < gap:
        return 1

    return 0


def peaks_and_valleys(ts, min_altitude_ratio=1e-3) -> Tuple:
    """求一维数组`ts`表示的的峰值和谷底，本函数可以辅助用以确定局部顶和底。


    使用order = 2即最相邻五个点来确认峰值和谷底。为平滑波动，事先对数组求移动平均值。

    `min_altitude_ratio`用来指定峰值与相邻点的海拔高度的最小值。小于这个最小值的，将不被认为是峰值。在这里，相邻点可能是峰值（或者谷底）右侧的1~3个点，也就是信号可能晚最多3个周期才能确认。

    ![](https://images.jieyu.ai/images/202111/20211108182305.png)

    Examples:
        >>> ts = np.array([3589.86, 3586.2 , 3587.35, 3587.  , 3590.6 , 3593.53, 3602.47,
                    3603.62, 3595.87, 3582.53, 3587.56, 3594.78, 3596.02, 3591.88,
                    3586.08, 3598.18, 3593.5 , 3587.3 , 3584.57, 3582.6 , 3588.16,
                    3586.72, 3592.24, 3596.06, 3593.38, 3597.39, 3603.25, 3609.86,
                    3610.58, 3618.01, 3615.55, 3612.88, 3603.94, 3597.5 , 3599.46,
                    3597.64, 3575.2 , 3565.64, 3559.96, 3564.71, 3561.38, 3559.98,
                    3555.47, 3562.31, 3535.2 , 3528.66, 3532.11, 3529.  , 3524.13,
                    3523.6 , 3520.83, 3518.42, 3505.15, 3515.32, 3525.08, 3523.94,
                    3531.4 , 3540.49, 3544.02, 3547.34, 3540.21, 3539.08, 3549.08,
                    3549.93, 3555.34, 3545.89, 3546.1 , 3544.48, 3554.43, 3548.42,
                    3537.14, 3522.33, 3477.68, 3482.24, 3499.03, 3505.63, 3497.15,
                    3501.23, 3498.22, 3492.46, 3484.18, 3482.13, 3502.18, 3498.54,
                    3515.66, 3514.5 , 3523.32, 3521.07, 3526.13, 3520.53, 3524.83,
                    3526.87, 3518.77, 3522.16, 3514.98, 3518.57, 3506.63, 3506.4 ,
                    3501.08, 3491.57], dtype=np.float32)
        >>> peaks_and_valleys(ts)
        (array([ 8, 15, 31, 43, 68, 78, 91]), array([13, 21, 53, 67, 81]))

    Args:
        ts ([type]): [description]
        min_altitude_ratio ([type], optional): [description]. Defaults to 1e-3.
        ma ([type], optional): [description]. Defaults to None.

    Returns:
        [Tuple]: indices of valleys and peaks
    """

    ma = moving_average(ts, 5)

    local_ma = argrelextrema(ma, np.greater, order=2)[0]
    local_mi = argrelextrema(ma, np.less, order=2)[0]

    if len(local_ma):
        local_ma += 4
    if len(local_mi):
        local_mi += 4

    peaks = set()
    # the peaks are calced by ma, so we need to adjust
    for ppeak in local_ma:
        if ppeak < 5:
            peaks.add(ppeak)
            continue

        view = ts[ppeak - 5 : ppeak + 5]
        pmax = np.argmax(view)
        vmax = np.max(view)

        if pmax not in [0, len(view) - 1]:
            lmin = np.min(view[:pmax])
            rmin = np.min(view[pmax + 1 :])
            if (
                vmax / lmin - 1 > min_altitude_ratio
                and vmax / rmin - 1 > min_altitude_ratio
            ):
                peaks.add(ppeak + pmax - 5)
        else:
            peaks.add(ppeak)

    valleys = set()
    for pvalley in local_mi:
        if pvalley < 5:
            valleys.append(pvalley)
            continue
        view = ts[pvalley - 5 : pvalley + 5]
        pmin = np.argmin(view)
        vmin = np.min(view)

        if pmin not in [0, len(view) - 1]:
            lmax = np.max(view[:pmin])
            rmax = np.max(view[pmin + 1 :])
            if (
                lmax / vmin - 1 > min_altitude_ratio
                and rmax / vmin - 1 > min_altitude_ratio
            ):
                valleys.add(pvalley + pmin - 5)
        else:
            valleys.add(pvalley)

    return sorted(peaks), sorted(valleys)


def dark_cloud_cover(bars, penetration=-0.025):
    """乌云盖顶，即日线高开大阴线，并且收盘价向下插入到前一日实体内

    Returns:
        -1 if there's not dark cloud cover, otherwise frames since signal fired
    """
    open_ = bars["open"]
    close = bars["close"]

    pos = np.argwhere(
        (close[:-1] > open_[:-1])
        & (open_[1:] > close[:-1])
        & (close[1:] / close[:-1] - 1 < penetration)
    ).flatten()

    if len(pos):
        return len(bars) - 1 - (pos + 1)

    return [-1]


def hammer(bars, shadow_length=0.03) -> Tuple:
    """锤头

    锤头是指日线开盘后，股价下行，随后又被强烈的买盘推至开盘价上方，形成一个带长下影线的阳线实体。
    https://www.investopedia.com/articles/active-trading/062315/using-bullish-candlestick-patterns-buy-stocks.asp

    返回结果包含了下影线的长度（通过百分比度量）
    """
    open_ = bars["open"]
    close = bars["close"]
    low = bars["low"]

    shadow = open_ / low - 1

    if len(bars.shape) == 0:
        if open_ < close and shadow > shadow_length:
            return True, shadow

        return None, None

    pos = np.argwhere((open_ < close) & (shadow > shadow_length)).flatten()
    return pos, shadow[pos]


def inverted_hammer(bars, shadow_length=0.03) -> Tuple:
    """倒锤头，即仙人指路。当出现在低位时，可能意味着主力将要发动进攻。

    需要注意的是，从更短周期k线来看，上攻时需要带量，否则只是个别散户的游戏，没有意义。

    https://www.investopedia.com/articles/active-trading/062315/using-bullish-candlestick-patterns-buy-stocks.asp

    Args:
        bars ([type]): [description]
        shadow_length (float, optional): [description]. Defaults to 0.03.
    """
    open_ = bars["open"]
    close = bars["close"]
    high = bars["high"]

    shadow = high / open_ - 1

    # 如果传入的是标量，即仅一根线
    if len(bars.shape) == 0:
        if open_ < close and shadow > shadow_length:
            return True, shadow

        return None, None

    pos = np.argwhere((open_ < close) & (shadow > shadow_length)).flatten()
    return pos, shadow[pos]


def parabolic_features(ts, rng=7, ma_win=5, calc_ma=True):
    """检测`ts`代表的最后7个周期的均线中，是否存在抛物线特征。"""
    if calc_ma:
        ts = moving_average(ts, ma_win)

    ts_ = ts[-rng:]
    (a, b, c), pmae = polyfit(ts_)

    # predict till next frame
    y_ = np.polyval((a, b, c), np.arange(rng + 1))

    # uncomment this to draw the lines
    # plt.plot(np.arange(len(ts)-rng, len(ts)), y_)
    # plt.plot(ts)

    vx = round(-b / (2 * a), 1)

    next_ts = reverse_moving_average(y_, rng, ma_win)
    pred_roc = next_ts / ts[-1] - 1

    dist = rng - vx

    return np.sign(pred_roc), pred_roc, dist, round(a, 4), round(pmae, 5)


def reversal_features(
    code, bars, frame_type: FrameType, ma=None, wr_win=60, peak_altitude=1e-3
):
    # 顺序
    # 0. WR
    # 1. RSI
    # 2. parabolic flip
    # 3. stock pattern
    # 4. local maximum/minimum
    from alpha.core.rsi_stats import rsi30, rsiday

    assert frame_type in [FrameType.DAY, FrameType.MIN30]
    features = []

    open_, high, low, close = bars["open"], bars["high"], bars["low"], bars["close"]

    if ma is None:
        ma = moving_average(bars["close"], 5)

        bars = bars[4:]
        open_, high, low, close = bars["open"], bars["high"], bars["low"], bars["close"]

    # wr
    hh = np.max(high[-wr_win:])
    ll = np.min(low[-wr_win:])
    wr = 1 - (hh - close[-1]) / (hh - ll)

    features.append(wr)

    # rsi and prsi
    rsi = relative_strength_index(close, period=6)[-3:]

    rsistats = rsi30 if frame_type == FrameType.MIN30 else rsiday
    prsi = [rsistats.get_proba(code, v) or -1 for v in rsi]

    features.extend(prsi)
    features.extend(rsi)

    # parabolic features
    features.extend(parabolic_features(close))

    # stock pattern
    # double bottom | hammer | invert_hammer| long down shadow | long up shadow | double top | darkcloud cover
    db = double_bottom(bars)

    flag, shadow = hammer(bars[-1])
    hm = shadow if flag else 0

    flag, shadow = inverted_hammer(bars[-1])
    ihm = shadow if flag else 0

    ds = down_shadow(bars[-1])
    us = up_shadow(bars[-1])

    dt = double_top(bars)

    dcc = dark_cloud_cover(bars[-2:])[0]

    # local maximum/minimum
    n = 10
    peaks, valleys = peaks_and_valleys(close[-n:], min_altitude_ratio=peak_altitude)
    peak = n - peaks[-1] if len(peaks) else -1
    valley = n - valleys[-1] if len(valleys) else -1

    features.extend([db, hm, ihm, ds, us, dt, dcc, peak, valley])

    columns = [
        "wr",
        "prsi_2",
        "prsi_1",
        "prsi_0",
        "rsi_2",
        "rsi_1",
        "rsi_0",
        "parab_flag",
        "parab_pred_roc",
        "parab_vx",
        "parab_a",
        "parab_pmae",
        "double_bottom",
        "hammer",
        "inverted_hammer",
        "down_shadow",
        "upper_shadow",
        "double_top",
        "dark_cloud_cover",
        "peak",
        "valley",
    ]
    return features, columns


def divergency(indicator, price, check_win=40) -> int:
    """检测指标与价格之间的背离

    如果指标在相邻的位置持续走低（或者走高），只保留最后一个位置上的数据进行计算。

    最多检测两次同方向背离。注意传入的指标和price都必须为正的序列。如果不满足此条件，需要进行预处理。
    如果返回2，表明发生两次底背离（价格仍在走低，但指标并未跟随走低）；如果返回-2，则表明发生两次顶背离（价格仍在走高，但指标并未跟随走高）。返回1和-1同理，但意味着仅发生一次背离。
    Returns:
        int: 背离次数（正负号表示方向），背离时指标位置
    """
    indicator = filterna(indicator)
    price = filterna(price)

    cw = check_win
    min_len = min(len(indicator), len(price), cw)
    indicator = indicator[-min_len:]
    price = price[-min_len:]

    indice = top_n_argpos(-indicator, min(10, len(indicator)))

    # 对连续出现的坐标，只保留最后一个
    pos = []

    pos_ = sorted(indice, reverse=True)
    for i, p in enumerate(pos_):
        if i == 0:
            pos.append(p)
            continue
        if p != pos_[i - 1] - 1:
            pos.append(p)

    # 最多检测两次背离
    if len(pos) > 3:
        pos = pos[:3]

    ind = indicator[pos]
    pri = price[pos]

    # later time positioned first
    if np.all(np.diff(ind) < 0) and np.all(np.diff(pri) > 0):
        return len(pos) - 1, pos
    if np.all(np.diff(ind) > 0) and np.all(np.diff(pri) < 0):
        return -len(pos) + 1, pos

    return 0, None


def long_parallel(arrays) -> Tuple[bool, int]:
    """检测二维数组`arrays`是否为多头排列

    `arrays`可以为二维numpy数组，也可以是二维Python数组。

    返回值: (是否为多头排列, 多头排列的长度)
    """
    arrays = np.array(arrays)

    flags = np.repeat(True, len(arrays[0]))
    rows = len(arrays)

    for i in range(rows - 1):
        flags &= arrays[i] >= arrays[i + 1]

    flags, _, lengths = find_runs(flags)
    return flags[-1], lengths[-1]


def short_parallel(arrays) -> Tuple[bool, int]:
    """检测二维数组`arrays`是否为多头排列

    `arrays`可以为二维numpy数组，也可以是二维Python数组。

    返回值: (是否为多头排列, 多头排列的长度)
    """
    arrays = np.array(arrays)

    flags = np.repeat(True, len(arrays[0]))
    rows = len(arrays)

    for i in range(rows - 1, 0, -1):
        flags &= arrays[i] >= arrays[i - 1]

    flags, _, lengths = find_runs(flags)
    return flags[-1], lengths[-1]


def altitude(bars: np.ndarray) -> float:
    """计算收盘价在序列中的高度，类似于wr指标

    返回值接近1时，表示收盘价越接近前高。当返回值为1时，意味着正在创新高。
    返回值接近0时，表示收盘价越接近前低。当返回值为0时，意味着正在创新低。
    """
    hh = np.max(bars["high"])
    ll = np.min(bars["low"])

    close = bars["close"]

    return 1 - (hh - close[-1]) / (hh - ll)

