import numpy as np

from alpha.core.features import fillna, replace_zero, top_n_argpos


def top_volume_direction(bars: np.array, n=21):
    """计算`n`周期内，较大成交量的方向和比值。

    成交量方向：如果当前股价上涨则成交量方向为1，下跌则为-1
    计算方法：
        1. 找出最大成交量的位置
        2. 找出其后一个最大异向成交量的位置
        3. 总共返回一个2维数组，0表示不存在（不适用）

    args:
        bars: 包含了OHLC和volume的行情数据，类型为numpy structured array
        n: 参与计算的周期。太长则影响到最大成交量的影响力。

    """
    bars = bars[-n:]
    volume = bars["volume"]

    flags = np.select((bars["close"] >= bars["open"],), [1], -1)

    vmin = np.min(volume)
    pmax = np.argmax(volume)

    vr = [volume[pmax] / vmin, 0]

    vol = (volume * flags)[pmax:]
    vmax = vol[0]

    if flags[pmax] == 1 and np.any(vol[1:] < 0):
        vr[1] = np.min(vol) / vmax
    elif flags[pmax] == -1 and np.any(vol[1:] > 0):
        vr[1] = np.max(vol) / vmax

    return vr
