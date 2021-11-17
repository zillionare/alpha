from typing import List
from alpha.strategies.z05 import Z05
from alpha.core import Frame
from omicron.core.timeframe import tf
from omicron.models.securities import Securities
from omicron.models.security import Security
import arrow
from omicron.core.types import FrameType
import numpy as np
from alpha.utils import first_mdd_less_than_threshold
from alpha.core.features import (
    filterna,
    ma_d1,
    maline_support_ratio,
    moving_average,
    relative_strength_index,
    rolling,
)
from alpha.core.rsi_stats import rsiday, rsi30
import logging

logger = logging.getLogger(__name__)


class Z51(Z05):
    """类似于Z05,但通过使用预先计算的指标，使回测速度更快

    Args:
        Z05 ([type]): [description]
    """

    def __init__(
        self,
        holding_days: int = 5,
        rsi=88,
        rsi3=90,
        prsi=0.9,
        msr=0.85,
        bcr=0.85,
        d1=0.01,
        stop_loss=-0.05,
    ):
        super().__init__(
            holding_days=holding_days,
            rsi=rsi,
            rsi3=rsi3,
            prsi=prsi,
            msr=msr,
            bcr=bcr,
            d1=d1,
            stop_loss=stop_loss,
        )
        self.xlen = 7

    async def backtest(self, start: Frame, end: Frame, code: str):
        start = arrow.get(start)
        end = arrow.get(end)

        ma_win = 5
        rsi_precharge = 6 * 3
        nbars = self.xlen + ma_win + rsi_precharge - 1

        start_ = tf.day_shift(start, -nbars)
        sec = Security(code)
        bars = await sec.load_bars(start_, end, FrameType.DAY)

        # filter out nan bars
        mask = np.argwhere(~np.isnan(bars["close"])).flatten()
        bars = bars[mask]
        nbars = len(bars)

        self.bars = bars
        close = bars["close"]

        aligned = max(rsi_precharge, ma_win) - nbars

        # 计算rsi和概率
        self.rsi = relative_strength_index(bars["close"])[aligned:]
        self.prsi = np.array([rsiday.get_proba(code, r) for r in self.rsi])
        good_rsi = (np.isnan(self.prsi[0]) & (self.rsi < self.rsi)) | (
            (~np.isnan(self.prsi[0])) & (self.prsi < self.prsi)
        )

        if not np.isnan(self.prsi[0]):
            good_rsi = self.prsi < self.prsi
        else:
            good_rsi = self.rsi < self.rsi

        # 计算ma5 和 moving average ma5 supported bars
        ma5 = moving_average(close, ma_win)
        supported_bars = np.select([close[ma_win - 1 :] >= ma5], [1], 0)
        msr = moving_average(supported_bars, self.feat_len)[aligned:]

        # 计算moving average bullish bars ratio
        bull_bars = np.select([close >= bars["open"]], [1], 0)
        mbr = moving_average(bull_bars, self.feat_len)[aligned:]

        # 计算上涨加速度
        macc = moving_average(close[1:] / close[:-1] - 1, 5)[aligned:]

        for i in range(3, abs(aligned)):
            if all(
                [
                    np.all(good_rsi[i - 3 : i + 1]),
                    msr[i] > self.msr,
                    mbr[i] > self.bcr,
                    macc[i] > self.d1,
                ]
            ):
                self.orders.append(
                    {
                        "name": sec.display_name,
                        "code": sec.code,
                        "buy_pos": nbars + aligned + i,
                        "buy_at": bars["frame"][nbars + aligned + i],
                        "status": "opened",
                        "buy": bars["close"][nbars + aligned + i],
                        "params": {
                            "rsi": self.rsi[i],
                            "prsi": self.prsi[i],
                            "msr": msr[i],
                            "mbr": mbr[i],
                            "macc": macc[i],
                        },
                    }
                )

        for order in self.orders:
            if order["status"] == "opened":
                pos = order["buy_pos"]
                self.close_order(order, bars[pos + 1 : pos + self.holding_days])

        return self.backtest_summary(start, end)

    def close_order(self, order, bars):
        """平仓

        如果期间触发操作信号，以触发信号的收盘价为卖价；否则，以停损时的收盘价为卖价，或者最后一个周期的收盘价为卖价

        Args:
            order ([type]): [description]
            bars ([type]): [description]
        """
        code = order["code"]
        buy_price = order["buy"]

        try:
            close = filterna(bars["close"])
            with_buy_price = np.insert(close, 0, buy_price)

            sell, i = first_mdd_less_than_threshold(with_buy_price, self.stop_loss)
            if sell is not None:
                isell = np.argwhere(bars["close"] == sell)[0][0]
                close_type = "mdd"
            else:
                isell = -1
                close_type = "expired"
        except Exception as e:
            logger.info("failed to close order of %s", code)
            return

        sell_price = bars["close"][isell]
        gains = sell_price / buy_price - 1
        sell_at = bars["frame"][isell]
        order.update(
            {
                "sell": sell_price,
                "sell_at": sell_at,
                "gains": gains,
                "duration": tf.count_day_frames(order["buy_at"], sell_at),
                "type": close_type,
                "status": "closed",
            }
        )
