import numpy as np
import math
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

def pos_encode(stationary: np.array, var: np.array)->float:
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
    spectrum = np.sin([1/(3**i) for i in range(len(stationary))])
    maxium = (stationary[::-1] - stationary).dot(spectrum)
    diff = var - stationary
    return diff.dot(spectrum) / maxium
