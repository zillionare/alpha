import logging
import os
from random import sample
from typing import List, NewType
from alpha.core import Frame
import pickle

import arrow
import cfg4py
import numpy as np
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.security import Security
from omicron.models.securities import Securities
from alpha.core.rsi_stats import rsi30, rsiday

from alpha.core.features import (
    bullish_candlestick_ratio,
    fillna,
    ma_d1,
    ma_d2,
    ma_permutation,
    maline_support_ratio,
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
        2. 不存在前一周期rsi超标、当前周期rsi超标以及上涨3%后rsi超标情况。rsi指标处于90%高位为超标
    平仓条件：
        1. rsi 30分钟线高位,或
        2. -5%止损，或
        3. 30分钟两次上攻失败,或
        4. 5天内两次上攻失败
    """

    def __init__(self, holding_days: int = 5):
        """
        Initializing the strategy
        """
        self.stop_lose = 0.95
        self.feat_len = 7
        self.xlen = self.feat_len + 5

        self.holding_days = holding_days
        self.ylen = holding_days * 8
        self.long_orders = []
        self.trades = []

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
        msr = maline_support_ratio(bars["close"], 5, self.feat_len)
        bcr = bullish_candlestick_ratio(bars, self.feat_len)
        rb = real_body(bars[-self.feat_len:])
        d1 = ma_d1(bars["close"], 5)[-1]
        d2 = ma_d2(bars["close"], 5)[-1]

        close = fillna(bars["close"].copy())
        rsi = relative_strength_index(close, period=6)[-1]
        prsi = rsiday.query(code, rsi)

        # 3 top singed volume out of 8 frames
        tvd = top_volume_direction(bars)

        # 最近三天量的变化情况，取一阶差分和（方向）及二阶导（加速度）
        volume = np.array(list(filter(lambda x: np.isfinite(x), bars["volume"]))[-3:])
        vr = volume[1:] / volume[:-1] - 1

        vr_mean = np.sum(vr)/2

        return (msr, bcr, rb, d1, d2, prsi, *tvd, vr_mean)

    def try_open_position(self, code: str, bars: np.array):
        features = self.extract_features(code, bars)
        if self.model:
            label, p = self.model.predict_proba([features])
            if label == 1 and p > 0.5:
                self.long_orders.append(
                    {
                        "code": code,
                        "order_date": bars["frame"][-1],
                        "buy_price": bars["close"][-1],
                        "params": (*features, p),
                    }
                )
        else:
            msr, bcr, rb, d1, d2, prsi, *_ = features
            if msr >= 0.5 and bcr >= 0.5 and d1 > 0 and prsi < 0.9:
                self.long_orders.append(
                    {
                        "code": code,
                        "order_date": bars["frame"][-1],
                        "buy_price": bars["close"][-1],
                        "params": (*features, 1),
                    }
                )

    def close_order(self, order, bars):
        """平仓

        如果期间触发操作信号，以触发信号的收盘价为卖价；否则，以停损时的收盘价为卖价，或者最后一个周期的收盘价为卖价

        Args:
            order ([type]): [description]
            bars ([type]): [description]
        """
        code = order["code"]
        buy_price = order["buy_price"]
        params = order["params"]

        close = bars["close"]

        isell = -1
        close_type = "expired"
        i_stop_lose = np.argwhere(close < buy_price * self.stop_lose).flatten()
        if len(i_stop_lose) > 0:
            isell = i_stop_lose[0]
            close_type = "stop_loss"
        try:
            rsi = relative_strength_index(bars["close"], period=6)
            pos = np.argmax(rsi)
            max_rsi = np.max(rsi)

            prsi = rsi30.query(code, max_rsi)
            if prsi >= 0.9:
                isell = pos + 6
                params["sell_rsi"] = max_rsi
                params["sell_prsi"] = prsi
                close_type = "rsi"
        except Exception as e:
            logger.exception(e)

        sell_price = bars["close"][isell]
        gains = sell_price / buy_price - 1
        sell_at = bars["frame"][isell]
        self.trades.append(
            {
                "code": code,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "gains": gains,
                "order_date": order["order_date"],
                "sell_at": sell_at,
                "close_type": close_type,
                "params": params,
            }
        )

    async def backtest(self, start: Frame, end: Frame, stocks: List = None):
        test_start = tf.day_shift(arrow.get(start), 0)
        test_end = tf.day_shift(arrow.get(end), 0)

        stocks = stocks or Securities().choose(["stock"])
        for frame in tf.get_frames(test_start, test_end, FrameType.DAY):
            bars_end = tf.int2date(frame)
            bars_start = tf.day_shift(bars_end, -self.xlen + 1)

            for code in stocks:
                sec = Security(code)
                try:
                    bars = await sec.load_bars(bars_start, bars_end, FrameType.DAY)
                except Exception:
                    continue

                close = bars["close"]
                if np.count_nonzero(np.isfinite(close)) < self.xlen * 0.9:
                    continue

                self.try_open_position(code, bars)

        # to test if we can close the orders
        for order in self.long_orders:
            code = order["code"]
            order_date = order["order_date"]

            ystart = tf.day_shift(order_date, 1)
            ystart = tf.combine_time(ystart, 10)

            yend = tf.shift(ystart, self.ylen - 1, FrameType.MIN30)

            sec = Security(code)
            ybars = await sec.load_bars(ystart, yend, FrameType.MIN30)
            self.close_order(order, ybars)

        return self.backtest_summary(test_start, test_end)

    def backtest_summary(self, start, end):
        assert len(self.long_orders) == len(self.trades)

        total_duration = (arrow.get(end) - arrow.get(start)).days

        ntrades = len(self.trades)
        exposure_time = 0
        returns = 0
        win_trades = 0

        best_trade = 0
        worst_trade = 0

        for trade in self.trades:
            trade_duration = (arrow.get(trade["sell_at"]) - arrow.get(trade["order_date"])).days
            exposure_time += trade_duration

            gains = trade["gains"]
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
            "avg_gains": returns / ntrades,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "win_rate": win_trades / ntrades,
            "exposure_time": exposure_time
        }
