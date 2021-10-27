import numpy as np

from alpha.core.features import fillna, replace_zero, top_n_argpos


def top_volume_direction(bars: np.array, n=8):
    """计算`n`周期内，最大3次成交量的方向和量能变化的幅度

    成交量方向：如果当前股价上涨则成交量方向为1，下跌则为-1

    Note: 8 is one of fabnaci numbers

    args:
        bars: 包含了OHLC和volumne的行情数据，类型为numpy structured array
        n: 参与计算的周期。太长则影响到最大成交量的影响力。

    """
    bars_ = bars[-n:]

    close = fillna(bars_["close"].copy())
    open_ = fillna(bars_["open"].copy())
    volume = bars_["volume"]

    avg = np.nanmean(volume)
    volume = replace_zero(volume.copy())

    # 涨跌，假定close > open,意味着方向为买进
    flags = np.where((close > open_), 1, -1)

    vr = volume / avg
    indice = top_n_argpos(vr, 3)

    # 加上方向
    vr = vr * flags

    # 按时间先后排列
    indice = np.sort(indice)
    return vr[indice]
