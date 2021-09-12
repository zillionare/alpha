import itertools
import math
from typing import Any, List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

argpos_permutations = {
    n: list(itertools.permutations(range(n))) for n in (2, 3, 4, 5, 6, 7)
}


def polyfit(ts: np.array, degree: int = 2) -> tuple:
    """fit ts with np.polyfit, return coeff and pmae"""
    coeff = np.polyfit(np.arange(len(ts)), ts, degree)
    pmae = np.abs(np.polyval(coeff, np.arange(len(ts))) - ts).mean() / np.mean(ts)
    return coeff.tolist(), pmae


def reverse_moving_average(ma: ArrayLike, i: int, win: int) -> float:
    """given moving_average, reverse the origin value at index i

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
    ts: ArrayLike, win: int, n_preds: int = 1, err_threshold=1e-2
) -> float:
    """predict the next ith value by fitted moving average

    make sure ts is not too long and not too short

    Args:
        ts (np.array): the time series
        i (int): the index of the value to be predicted, start from 1
        win (int): the window size

    Returns:
        float: the predicted value
    """
    ma = moving_average(ts, win)

    n = 7 if win == 5 else 20

    if len(ma) < n:
        raise ValueError(f"{len(ma)} < {n}, can't predict")

    ma = ma[-n:]
    coef, pmae = polyfit(ma, degree=2)
    if pmae > err_threshold:
        return None, None

    fitma = np.polyval(coef, np.arange(len(ma) + n_preds))
    preds = [reverse_moving_average(fitma, i, win) for i in range(len(fitma))]

    return preds[-n_preds:], pmae


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


def relative_strength_index(prices, period=14):
    """
    The Relative Strength Index (RSI) is a momentum oscillator.
    It oscillates between 0 and 100.
    It is considered overbought/oversold when it's over 70/below 30.
    Some traders use 80/20 to be on the safe side.
    RSI becomes more accurate as the calculation period (min_periods)
    increases.
    This can be lowered to increase sensitivity or raised to decrease
    sensitivity.
    10-day RSI is more likely to reach overbought or oversold levels than
    20-day RSI. The look-back parameters also depend on a security's
    volatility.

    Like many momentum oscillators, overbought and oversold readings for RSI
    work best when prices move sideways within a range.

    You can also look for divergence with price.
    If the price has new highs/lows, and the RSI hasn't, expect a reversal.
    Signals can also be generated by looking for failure swings and centerline
    crossovers.

    RSI can also be used to identify the general trend.

    The RSI was developed by J. Welles Wilder and was first introduced in his
    article in the June, 1978 issue of Commodities magazine, now known as
    Futures magazine. It is detailed in his book New Concepts In Technical
    Trading Systems.

    http://www.csidata.com/?page_id=797
    http://stockcharts.com/help/doku.php?id=chart_school:technical_indicators:relative_strength_in

    Input:
      prices ndarray
      period int > 1 and < len(prices) (optional and defaults to 14)

    Output:
      rsis ndarray

    Test:

    >>> import numpy as np
    >>> prices = np.array([44.55, 44.3, 44.36, 43.82, 44.46, 44.96, 45.23,
    ... 45.56, 45.98, 46.22, 46.03, 46.17, 45.75, 46.42, 46.42, 46.14, 46.17,
    ... 46.55, 46.36, 45.78, 46.35, 46.39, 45.85, 46.59, 45.92, 45.49, 44.16,
    ... 44.31, 44.35, 44.7, 43.55, 42.79, 43.26])
    >>> print(rsi(prices))
    [ 70.02141328  65.77440817  66.01226849  68.95536568  65.88342192
      57.46707948  62.532685    62.86690858  55.64975092  62.07502976
      54.39159393  50.10513101  39.68712141  41.17273382  41.5859395
      45.21224077  37.06939108  32.85768734  37.58081218]
    """

    num_prices = len(prices)

    if num_prices < period:
        # show error message
        raise SystemExit("Error: num_prices < period")

    # this could be named gains/losses to save time/memory in the future
    changes = prices[1:] - prices[:-1]
    # num_changes = len(changes)

    rsi_range = num_prices - period

    rsis = np.zeros(rsi_range)

    gains = np.array(changes)
    # assign 0 to all negative values
    masked_gains = gains < 0
    gains[masked_gains] = 0

    losses = np.array(changes)
    # assign 0 to all positive values
    masked_losses = losses > 0
    losses[masked_losses] = 0
    # convert all negatives into positives
    losses *= -1

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsis[0] = 100
    else:
        rs = avg_gain / avg_loss
        rsis[0] = 100 - (100 / (1 + rs))

    for idx in range(1, rsi_range):
        avg_gain = (avg_gain * (period - 1) + gains[idx + (period - 1)]) / period
        avg_loss = (avg_loss * (period - 1) + losses[idx + (period - 1)]) / period

        if avg_loss == 0:
            rsis[idx] = 100
        else:
            rs = avg_gain / avg_loss
            rsis[idx] = 100 - (100 / (1 + rs))

    return rsis


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
