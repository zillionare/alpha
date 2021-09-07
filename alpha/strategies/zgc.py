
"""2021年9月初，中关村"""

from matplotlib.pyplot import close
from pyrsistent import v
from alpha.core.features import predict_by_moving_average, replace_zero
from typing import List
import arrow
import numpy as np
import pandas as pd
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security
import logging

logger = logging.getLogger(__name__)

class ZGCStrategy(object):
    """中关村"""
    def __init__(self) -> None:
        self.ma_wins = [5, 10, 20]
        self.nbars = 40
        super().__init__()

    def get_pmae_err_threshold(self, win, frame_type: FrameType = FrameType.MIN30):
        if frame_type == FrameType.MIN30:
            return {
                5: 3e-3,
                10: 1e-3,
            }.get(win, 1e-4)
        elif frame_type == FrameType.DAY:
            return {5: 8e-3, 10: 5e-3, 20: 3e-3}.get(win, 3e-3)

    async def scan(self, frame_type: FrameType, profit=0.15, codes:List=None, end=None):
        """
        刷新
        """
        results = []
        frame_type = FrameType(frame_type)

        if end is None:
            if frame_type == FrameType.DAY:
                end = arrow.now().date()
            else:
                end = tf.floor(arrow.now(), frame_type)

        codes = codes or Securities().choose(["stock"])
        for code in codes:
            try:
                sec = Security(code)

                start = tf.shift(end, -self.nbars + 1, frame_type)

                bars = await sec.load_bars(start, end, frame_type)
                if (
                    bars is None
                    or len(bars) != self.nbars
                    or np.count_nonzero(np.isnan(bars["close"])) > len(bars) * 0.1
                ):
                    continue

                close = bars["close"].copy()
                ypred = self.predict_profit(close, frame_type)
                if ypred is None or ypred < profit:
                    continue

                if self.slzl(bars):
                    print(sec.display_name, f"{ypred:.0%}")
                    results.append((sec.display_name, ypred))

            except Exception as e:
                logger.exception(e)
                continue
        return pd.DataFrame(results, columns=["name", "profit"])

    def predict_profit(self, close, frame_type, ylen=5):
        """
        预测盈利
        """
        ypred = 0
        for win in self.ma_wins:
            _ypreds, _ = predict_by_moving_average(
                close, win, 5, self.get_pmae_err_threshold(win, frame_type)
            )

            if _ypreds is None:
                continue

            # 如果长线看空，则不操作
            if _ypreds[-1] < close[-1] and win in [10, 20]:
                return None

            ypred = max(ypred, max(_ypreds))

        return ypred/close[-1] - 1

    def slzl(self, bars):
        """
        三周期内是否缩量整理
        """
        volume = replace_zero(bars["volume"].copy())[-21:]

        vcr = volume[1:]/volume[:-1]
        var = (volume/np.mean(volume))[-20:]

        open_ = bars["open"]
        close = bars["close"]

        flags = np.where((close > open_)[1:] & (close[1:] > close[:-1]), 1, -1)

        vcr = vcr[-3:]
        var = var[-3:]
        flags = flags[-3:]

        if (vcr[0] > 2 or var[0] > 5) and flags[0] == 1: # 三日前放量涨
            if np.all(vcr[1:] < 0.85) and np.all(flags[1:] == -1):
                # 后两日缩量跌
                return True
        return False

