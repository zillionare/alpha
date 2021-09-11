import numpy as np

from alpha.core.features import fillna, replace_zero, top_n_argpos


def top_volume_direction(bars: np.array, n=10):
    """计算`n`周期内，最大3次成交量的方向和量能变化的幅度

    成交量方向：如果当前股价上涨则成交量方向为1，下跌则为-1

    args:
        bars: 包含了OHLC和volumne的行情数据，类型为numpy structured array
        n: 参与计算的周期。太长则影响到最大成交量的影响力。

    """
    close = fillna(bars["close"].copy())
    open_ = fillna(bars["open"].copy())

    avg = np.nanmean(bars["volume"][-n:])
    volume = replace_zero(bars["volume"].copy())

    # 涨跌
    flags = np.where((close > open_)[1:] & (close[1:] > close[:-1]), 1, -1)

    vr = volume[-n:] / avg
    indice = top_n_argpos(vr, 3)

    # 加上方向
    vr *= flags[-n:]

    # 按时间先后排列
    indice = np.sort(indice)
    return vr[indice]
