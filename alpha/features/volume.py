from alpha.core.features import fillna, replace_zero, top_n_argpos
import numpy as np

class VolumeFeatures:
    """Features: Volume"""

    def __init__(self, features=None):
        self.features = features

    def __str__(self) -> str:
        return f"成交量特征"

    def transform(self, bars: np.array, n=10):
        """计算`n`周期内，最大3次成交量(且成交量大于成交均量的1倍）的方向和量能变化的幅度

        成交量方向：如果当前股价上涨则成交量方向为1，下跌则为-1
        如果存在上下影线，则成交量需要进行调整，调整方法是乘以实体比例。成交量方向叠加在量能变化幅度上。

        args:
            bars: 包含了OHLC和volumne的行情数据，类型为numpy structured array
            n: 参与计算的周期。太长则影响到最大成交量的影响力。

        """        
        close = fillna(bars["close"].copy())

        high = bars["high"].copy()
        low = bars["low"].copy()
        _open = bars["open"].copy()

        close = fillna(close.copy())
        high = fillna(high)
        low = fillna(low)
        _open = fillna(_open)

        avg = np.nanmean(bars["volume"][-n:])

        volume = replace_zero(bars["volume"].copy())

        # 涨跌
        flags = np.where((close > _open)[1:] & (close[1:] > close[:-1]), 1, -1)

        vr = volume[-n:] / avg
        indice = top_n_argpos(vr, 3)

        # 加上方向
        vr *= flags[-n:]

        # 按时间先后排列
        indice = np.sort(indice)
        return vr[indice]
