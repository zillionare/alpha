from typing import Tuple

import numpy as np

from omicron.extensions.np import rolling


def top_volume_direction(bars: np.array, n=10) -> Tuple[float, float]:
    """计算`n`周期内，较大成交量的方向和比值。

    成交量方向：如果当前股价上涨则成交量方向为1，下跌则为-1

    计算方法：
        1. 找出最大成交量的位置
        2. 找出其后一个最大异向成交量的位置
        3. 总共返回一个2维数组，0表示不存在（不适用）

    args:
        bars: 包含了OHLC和volume的行情数据，类型为numpy structured array, 长度应该大于`2 * n`
        n: 参与计算的周期。太长则影响到最大成交量的影响力。

    return:
        最大成交量量比及方向，最大成交之后，异向成交量与最大成交量的比值。0，0表示不适用（如当期为最大成交量，此后不存在异向成交量）

    """
    assert len(bars) >= 2 * n, "bars length should be greater than 2 * n"
    bars_ = bars

    bars = bars[-n:]
    volume = bars["volume"]

    flags = np.select((bars["close"] >= bars["open"],), [1], -1)

    pmax = np.argmax(volume)

    # 最大成交量前n个bar的均量
    vmean = np.mean(bars_["volume"][pmax - 2 * n : -n + pmax])

    vol = (volume * flags)[pmax:]
    vmax = vol[0]

    if flags[pmax] == 1 and np.any(vol[1:] < 0):
        vr = [vmax / vmean, np.min(vol) / vmax]
    elif flags[pmax] == -1 and np.any(vol[1:] > 0):
        vr = [vmax / vmean, abs(np.max(vol) / vmax)]
    else:
        vr = [vmax / vmean, 0]

    return vr


def moving_net_volume(bars, win=5) -> np.array:
    """移动净余成交量
    args:
        bars: 包含了OHLC和volume的行情数据，类型为numpy structured array

    return:
        np.array: `win`周期内归一化的的移动和

    """
    vol = bars["volume"]
    close = bars["close"]
    open_ = bars["open"]

    vol = vol / vol[0]

    flags = np.select((close > open_, close < open_), [1, -1], 0)
    signed_vol = vol * flags

    return rolling(signed_vol, win, np.sum)


def net_buy_volume(bars) -> float:
    """bars全部区间内的净买入量

    如果k线为阳线，则为买入量；如果为阴线，则为卖出量
    """
    volume = bars["volume"]
    volume = volume / volume[0]
    return np.sum(volume * np.sign(bars["close"] - bars["open"]))
