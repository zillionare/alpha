import logging

import numpy as np

from alpha.backtesting.strategy import Strategy
from omicron.core import talib

logger = logging.getLogger()


class Z01Strategy(Strategy):
    """A very basic policy based on volume

    Args:
        backtesting ([type]): [description]
    """

    def __init__(self, win: float = 8, f1: float = 0.4, f2: float = 1.4):
        self.win = win
        self.f1 = f1
        self.f2 = f2

        self.vol_sum_signed = self.I(self.calc_cum_vol)
        self.vol_average = self.I(talib.moving_average, self.data.Volume, self.win)

    def calc_cum_vol(self):
        signs = np.where(self.data.Close >= self.data.Open, 1, -1)
        signed_volume = self.data.Volume * signs

        return talib.rolling(signed_volume, self.win, "sum")

    def next(self):
        vol_sum_signed = self.vol_sum_signed[-2]
        vol_average = self.vol_average[-2]

        today = self.data.index[-2]
        if vol_sum_signed >= vol_average * self.f1 and self.position.size == 0:
            logger.info(">>> [%s] equitity: %s", today, self.equity)
            self.buy()
        elif vol_sum_signed <= -vol_average * self.f2 and self.position.size > 0:
            self.position.close()
            logger.info("<<< [%s], equitity:%s", today, self.equity)
