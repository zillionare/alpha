import logging

import numpy as np

from alpha.backtesting.strategy import Strategy
from omicron.core.talib import moving_average, rolling

logger = logging.getLogger()


class Z02Strategy(Strategy):
    """Strategy based on moving average

    Args:
        Strategy ([type]): [description]
    """

    win = 8
    f1 = 0.4
    f2 = 1.4

    def init(self):
        self.vol_sum_signed = self.declare_indicator(self.calc_cum_vol)
        self.vol_average = self.declare_indicator(
            moving_average, self.data.Volume, self.win
        )

    def calc_cum_vol(self):
        signs = np.where(self.data.Close >= self.data.Open, 1, -1)
        signed_volume = self.data.Volume * signs

        return rolling(signed_volume, self.win, "sum")

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
