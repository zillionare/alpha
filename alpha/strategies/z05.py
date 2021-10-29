import logging
import os
from random import sample
from typing import List, NewType
from alpha.core import Frame
import pickle
import time

import arrow
import cfg4py
import numpy as np
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.security import Security
from omicron.models.securities import Securities
from alpha.core.rsi_stats import rsi30, rsiday
from alpha.utils import buy_limit_price, equal_price
import asyncio

from alpha.core.features import (
    bullish_candlestick_ratio,
    fillna,
    ma_d1,
    ma_d2,
    ma_permutation,
    maline_support_ratio,
    moving_average,
    real_body,
    relative_strength_index,
    transform_y_by_change_pct,
)
from alpha.features.volume import top_volume_direction
from alpha.strategies.base_xgboost_strategy import BaseXGBoostStrategy
from alpha.utils.data import DataBunch

logger = logging.getLogger(__name__)

cfg = cfg4py.init()


class Z05(object):
    """策略： 5日均线顺上选股策略

    开仓条件：
        1. 股价沿5日均线上线
        2. 不存在当前周期rsi超标以及上涨3%后rsi超标情况。rsi指标处于90%高位为超标
    平仓条件：
        1. rsi 30分钟线高位,或
        2. -5%止损，或
        3. 30分钟两次上攻失败,或
        4. 5天内两次上攻失败
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
    ):
        """
        Initializing the strategy
        """
        self.stop_loss = 0.95
        self.rsi = rsi
        self.rsi3 = rsi3
        self.prsi = prsi
        self.msr = msr
        self.bcr = bcr
        self.d1 = d1

        self.feat_len = 7
        self.xlen = self.feat_len + 5

        self.holding_days = holding_days
        self.ylen = holding_days * 8
        self.orders = []

        model_file = os.path.expanduser("~/zillionare/aplha/z05.model.pkl")
        try:
            self.model = self.load_model(model_file)
        except Exception as e:
            logger.exception(e)
            self.model = None

    def load_model(self, model_file: str):
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

    def extract_features(self, code: str, bars: np.array) -> np.ndarray:
        close = fillna(bars["close"].copy())

        # 5日均线对股价的支撑率， 最后一周期股价是否在均线上？
        msr, cur_ms = maline_support_ratio(close, 5, self.feat_len)
        # 阳线比率
        bcr = bullish_candlestick_ratio(bars, self.feat_len)
        # 实体大小？
        rb = real_body(bars[-self.feat_len :])
        # 上涨速率？
        d1 = ma_d1(close, 5)[-1]
        d2 = ma_d2(close, 5)[-1]

        # rsi和rsi的概率
        future_close = np.append(close, close[-1] * 1.03)
        rsi, rsi3 = relative_strength_index(future_close, period=6)[-2:]

        prsi = rsiday.get_proba(code, rsi)
        prsi3 = rsiday.get_proba(code, rsi3)

        # 3 top singed volume out of 8 frames
        tvd = top_volume_direction(bars)

        # 最近三天量的变化情况，取一阶差分和（方向）及二阶导（加速度）
        volume = np.array(list(filter(lambda x: np.isfinite(x), bars["volume"]))[-3:])
        vr = volume[1:] / volume[:-1] - 1

        vr_mean = np.sum(vr) / 2

        return (msr, cur_ms, bcr, rb, d1, d2, rsi, prsi, rsi3, prsi3, *tvd, vr_mean)

    def predict(self, sec: Security, bars: np.array, buy_price: float = None):
        """判断是否存在买入信号

        Args:
            code (str): [description]
            bars (np.array): [description]
            buy_price (float, optional): [description]. Defaults to None.
        """
        code, name = sec.code, sec.display_name
        close = fillna(bars["close"].copy())

        # 如果当前收盘价已涨停，则无法下单
        c0, c1 = close[-2:]
        if c1 >= buy_limit_price(code, c0, bars["frame"][-1]):
            return

        features = self.extract_features(code, bars)
        (
            msr,
            cur_ms,
            bcr,
            rb,
            d1,
            d2,
            rsi,
            prsi,
            rsi3,
            prsi3,
            tvd1,
            tvd2,
            tvd3,
            vr_mean,
        ) = features

        buy_price = buy_price or close[-1]

        if self.model:
            label, p = self.model.predict_proba([features])
            if label == 1 and p > 0.5:
                self.orders.append(
                    {
                        "code": code,
                        "order_date": bars["frame"][-1],
                        "buy": buy_price,
                        "params": {
                            "msr": msr,
                            "bcr": bcr,
                            "rb": rb,
                            "d1": d1,
                            "d2": d2,
                            "prsi": prsi,
                            "tvd1": tvd1,
                            "tvd2": tvd2,
                            "tvd3": tvd3,
                            "vr_mean": vr_mean,
                        },
                    }
                )
        else:
            good_rsi = (
                (prsi and prsi <= self.prsi) or (prsi is None and rsi < self.rsi)
            ) and (
                (prsi3 and prsi3 <= self.prsi) or (prsi3 is None and rsi3 <= self.rsi3)
            )

            if (
                msr >= self.msr
                and cur_ms
                and bcr >= self.bcr
                and d1 > self.d1
                and good_rsi
            ):
                if self.get_order_status(code) == "opened":
                    return

                logger.info("order: %s，%s", name, bars["frame"][-1])
                self.orders.append(
                    {
                        "name": name,
                        "code": code,
                        "status": "opened",
                        "buy_at": bars["frame"][-1],
                        "buy": buy_price,
                        "params": {
                            "msr": msr,
                            "bcr": bcr,
                            "rb": rb,
                            "d1": d1,
                            "d2": d2,
                            "prsi": prsi,
                            "prsi3": prsi3,
                            "tvd1": tvd1,
                            "tvd2": tvd2,
                            "tvd3": tvd3,
                            "vr_mean": vr_mean,
                        },
                    }
                )

    def get_order_status(self, code: str) -> int:
        """
        获取订单状态

        Args:
            order (dict): 订单信息
            bars (np.array): 当前K线
        """
        for order in self.orders[::-1]:
            if order["code"] == code:
                return order["status"]

        return None

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
            close = fillna(bars["close"].copy())
        except Exception as e:
            logger.exception(e)
            logger.info("failed to close order of %s", code)
            return

        isell = len(bars) - 1
        i_stop_loss = None
        irsi = None

        close_type = "expired"
        stop_loss = np.argwhere(close < buy_price * self.stop_loss).flatten()
        if len(stop_loss) > 0:
            i_stop_loss = stop_loss[0]
            close_type = "stop_loss"

        try:
            rsi = relative_strength_index(bars["close"], period=6)
            prsi = np.array([rsi30.get_proba(code, r) for r in rsi])
            if np.any(prsi):
                pos_rsi = np.argwhere(prsi > self.prsi).flatten()
                if len(pos_rsi) > 0:
                    irsi = pos_rsi[0] + 6
            else:
                pos_rsi = np.argwhere(rsi > self.rsi).flatten()
                if len(pos_rsi) > 0:
                    irsi = pos_rsi[0] + 6

            if np.all([irsi is not None, i_stop_loss is not None]):
                if irsi < i_stop_loss:
                    close_type = "rsi"
                    isell = irsi
                else:
                    close_type = "stop_loss"
                    isell = i_stop_loss
            elif irsi is not None:
                close_type = "rsi"
                isell = irsi
            elif i_stop_loss is not None:
                close_type = "stop_loss"
                isell = i_stop_loss

        except Exception as e:
            logger.exception(e)

        sell_price = bars["close"][isell]
        gains = sell_price / buy_price - 1
        sell_at = bars["frame"][isell]
        order.update(
            {
                "sell": sell_price,
                "sell_at": sell_at,
                "gains": gains,
                "duration": (arrow.get(sell_at) - arrow.get(order["buy_at"])).days,
                "type": close_type,
                "status": "closed",
            }
        )

    async def backtest(self, start: Frame, end: Frame, stocks: List = None):
        t0 = time.time()
        test_start = tf.day_shift(arrow.get(start), 0)
        test_end = tf.day_shift(arrow.get(end), 0)

        stocks = stocks or Securities().choose(["stock"])
        for frame in tf.get_frames(test_start, test_end, FrameType.DAY):
            await self.scan(tf.int2date(frame), stocks)

        # to test if we can close the orders
        for order in self.orders:
            if order["status"] == "closed":
                continue

            code = order["code"]
            order_date = order["buy_at"]

            ystart = tf.day_shift(order_date, 1)
            ystart = tf.combine_time(ystart, 10)

            yend = tf.shift(ystart, self.ylen - 1, FrameType.MIN30)

            sec = Security(code)
            ybars = await sec.load_bars(ystart, yend, FrameType.MIN30)
            self.close_order(order, ybars)

        elpased = time.time() - t0
        logger.info("backtest cost %s seconds", elpased)
        return self.backtest_summary(test_start, test_end)

    def backtest_summary(self, start, end):
        total_duration = (arrow.get(end) - arrow.get(start)).days

        ntrades = 0
        exposure_time = 0
        returns = 0
        win_trades = 0

        best_trade = 0
        worst_trade = 0

        for order in self.orders:
            if order["status"] != "closed":
                logger.info("unclosed oder found: %s", order)
                continue

            ntrades += 1
            exposure_time += order["duration"]

            gains = order["gains"]
            returns += gains
            if gains > 0:
                win_trades += 1

            best_trade = max(best_trade, gains)
            worst_trade = min(worst_trade, gains)

        return {
            "start": start,
            "end": end,
            "duration": total_duration,
            "returns": returns,
            "avg_gains": returns / ntrades if ntrades > 0 else "NA",
            "win_trades": win_trades,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "win_rate": win_trades / ntrades if ntrades > 0 else "NA",
            "exposure_time": exposure_time,
            "total_trades": ntrades,
        }

    async def scan(self, frame: Frame, stocks: List = None):
        """扫描`stocks`列表中的股票，看它们在`frame`时间段内是否有操作信号

        如果发现操作信号，则把股票添加到`self.long_orders`中

        Args:
            frame (Frame): 截止时间
            stocks (List): 股票列表
        """
        stocks = stocks or Securities().choose(["stock"])

        bars_start = tf.day_shift(frame, -self.xlen + 1)

        tasks = []
        for code in stocks:
            tasks.append(self.predict_one(code, bars_start, frame))

        await asyncio.gather(*tasks)

    async def predict_one(self, code, bars_start, bars_end):
        try:
            sec = Security(code)
            bars = await sec.load_bars(bars_start, bars_end, FrameType.DAY)
            close = bars["close"]
            if np.count_nonzero(np.isfinite(close)) < self.xlen * 0.9:
                return

            self.predict(sec, bars)
        except Exception:
            pass


