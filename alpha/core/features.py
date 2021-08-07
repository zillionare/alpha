import math
from typing import List

import numpy as np
from numpy.typing import ArrayLike


def moving_average(ts: np.array, win: int):
    """计算时间序列ts在win窗口内的移动平均

    Example:

        >>> ts = np.arange(7)
        >>> moving_average(ts, 5)
        >>> array([2.0000, 3.0000, 4.0000])

    """

    return np.convolve(ts, np.ones(win) / win, "valid")


def fillna(ts: np.array):
    """将ts中的NaN替换为其前值

    Args:
        ts (np.array): [description]
    """
    if np.all(np.isnan(ts)):
        raise ValueError("all of ts are NaN")

    if ts[0] is None or math.isnan(ts[0]):
        idx = np.argmin(np.isnan(ts))
        ts[0] = ts[idx]

    mask = np.isnan(ts)
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

def ma_permutation(ts: ArrayLike, n_features: int, ma_groups: List[int]):
    """
    Args:
        bars (list): [description]
    """
    mas = np.array([moving_average(ts, n)[-n_features:] for n in ma_groups])

    stationary = np.array(ma_groups)

    codes = []
    for i in range(n_features):
        pos = np.argsort(mas[:, i])
        codes.append(pos_encode(stationary, stationary[pos]))

    return codes

def transform_by_advance(ts: np.array, watermarks: List[float]):
    """根据涨跌幅转换成为标签

    Args:
        ts ([type]): [description]
        bin_cuts ([type]): 用以分类的threshold，必须是长度为2的升序列表，如(0.95, 1.05)
    """
    assert len(ts) >= 2

    c0 = ts[0]
    if c0 == np.NaN or np.all(ts[1:] == np.NaN):
        return None

    # 止损优先
    if min(ts[1:]) / c0 <= watermarks[0]:
        return -1
    elif max(ts[1:]) / c0 >= watermarks[1]:
        return 1
    else:
        return 0
