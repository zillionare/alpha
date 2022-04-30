import asyncio
import datetime
import json
import logging
import os
import uuid
from typing import Dict
from wsgiref import headers
import requests
import arrow
import cfg4py
import fire
import numpy as np
import omicron
from coretypes import FrameType
from empyrical import max_drawdown
from omicron import tf, math_round
from omicron.models.stock import Stock
from traderclient.client import TradeClient

from alpha.config import get_config_dir
from alpha.core.commons import plateaus
from alpha.utils.data.securities import Securities
from .base import BaseStrategy

# url = "http://192.168.100.114:7080/backtest/api/trade/v0.2/"
url = "http://192.168.100.112:3180/backtest/api/trade/v0.2/"

logger = logging.getLogger(__name__)


def price_equal(a, b):
    return math_round(a, 2) == math_round(b, 2)


class BreakoutStrategy(BaseStrategy)
    def __init__(self, url: str):
        # token = uuid.uuid4().hex
        name = "breakout_strategy-v0"
        token = "aaron-breakout-token"
        self.cash = 1_000_000
        self.broker = TradeClient(url, name, token, is_backtest=True, capital=self.cash)

        self.positions = {}

        self.thresholds = {"mdd": 0.1, "sl": 0.05}

    def update_portifolio(self, date: datetime.date):
        """更新当前可用现金和持仓"""
        self.cash = self.broker.available_money()["data"]
        positions = self.broker.positions(date)["data"]
        for item in positions:
            code, _, sellable, price = (
                item["security"],
                item["shares"],
                item["sellable"],
                item["price"],
            )

            local = self.positions.get(code, None)
            if local is None:
                logger.warning("服务器持仓在本地不存在，code:%s", code)
                continue
            if item["security"] not in self.positions:
                # 本地已下单，但服务器撮合失败
                logger.info("本地持仓在服务器上不存在:%s", item["security"])
                del self.positions[code]

            local["price"] = price
            local["sl"] = price * (1 - self.thresholds["sl"])
            local["sellable"] = sellable

    async def handle_signals(self, signals: Dict):
        """将signals中的信号转换成交易指令并执行。

        signals是一个队列，其中的每个元素包含：
        - code: 股票代码
        - signal: 信号，BUY/SELL
        - price: 价格,如果没有，则市价买入
        - score: 信号质量，越大越好，决定了哪些买入信号被优先执行。
        - time: 信号发出的时间

        执行时，对卖出信号，立即执行；如果是买入信号，则按取开盘前半小时个股分仓买入。
        Args:
            signals : _description_
        """
        pass

    async def run(self, start: str, end: str):
        """从`start`到`end`按天运行策略。

        在每天运行前，进行持仓更新，收集策略发出的评估信号，根据持仓和现金执行交易。
        """
        await omicron.init()

        start = tf.day_shift(arrow.get(start), 0)
        end = tf.day_shift(arrow.get(end).date(), 0)

        while start <= end:
            logger.info("evaluating %s", start)
            self.update_portifolio(start)
            await self.scan(start)
            start = tf.day_shift(start, 1)

        headers = {"Authorization": self.broker.token}
        bills = requests.get(url + "bills", headers=headers).json()
        with open("/tmp/breakout.bills", "w") as f:
            json.dump(bills, f)

        print(self.broker.metrics())

    async def evaluate_short(self, code: str, day: datetime.date):
        """是否需要卖出股票

        卖出条件：
            1. 跌破突破价，卖出
            2. 止损卖出
            3. 最大回撤卖出
            4. 滞胀卖出
        """
        # 如果现价低于买入时的条件，则平仓，等待下一次机会
        pos = self.positions[code]

        bars = await Stock.get_bars(code, 5, FrameType.DAY, day)
        close = bars["close"]
        c0 = close[-1]
        returns = close[1:] / close[:-1]
        mdd = max_drawdown(returns)

        order_time = tf.combine_time(bars[-1]["frame"], 14, 57)

        if mdd >= self.thresholds["mdd"]:
            logger.info("%s 卖出，mdd:%s", code, mdd)
            self._sell(code, pos["sellable"], order_time)
            del self.positions[code]
            return

        if c0 < pos["breakout_price"]:
            logger.info("模式卖出%s，现价:%s, 突破价:%s", code, c0, pos["breakout_price"])
            self._sell(code, pos["sellable"], order_time)
            del self.positions[code]
            return

        if c0 < pos["sl"]:
            logger.info("止损卖出%s, 现价:%s, 止损价:%s", code, c0, pos["sl"])
            return

        # 滞涨卖出: 当前价格未突破5日平台
        if c0 <= np.mean(close[-5:]) + 3 * np.std(close[-5:]):
            logger.info("滞涨卖出%s", code)
            self._sell(code, pos["sellable"], order_time)
            return

    async def scan(self, tm: datetime.date):
        """
        扫描指定时间段内的股票代码
        """
        tm = arrow.get(tm).date()

        secs = Securities.get_instance()
        codes = secs.query(tm).types(["stock"]).exclude_exit(tm).exclude_st().codes

        for code in codes:
            if self.cash > 10_000 and code not in self.positions:
                await self.evaluate_long(code, tm)
            if code in self.positions and self.positions[code].get("sellable", 0) > 0:
                await self.evaluate_short(code, tm)

    def _buy(self, code, shares, order_time: datetime.datetime, price=None):
        self.broker.buy(
            code, price, shares, order_time=order_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.update_portifolio(order_time.date())

    def _sell(self, code, shares, order_time: datetime.datetime, price=None):
        self.broker.sell(
            code, price, shares, order_time=order_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.update_portifolio(order_time.date())

    async def evaluate_long(self, code: str, day: datetime.date):
        bars = await Stock.get_bars(code, 100, FrameType.DAY, day)

        if bars.size < 20:
            return

        close = bars["close"]
        frame = bars["frame"]
        c1, c0 = close[-2:]

        platforms = plateaus(close[:-1], 20)
        if len(platforms) > 0:
            i, width = platforms[-1]
            plat_high = np.max(bars["high"][i : i + width])
            if c0 > plat_high and c1 < plat_high:
                logger.info(
                    "%s平台突破(%s~%s): %s long, 突破日%s",
                    code,
                    frame[i],
                    frame[i + width - 1],
                    width,
                    frame[-1],
                )
                shares = self.cash // c0 // 100
                logger.info("买入%s, 现价:%s, 突破价:%s", code, c0, plat_high)

                self.positions[code] = {"code": code, "breakout_price": plat_high}
                order_time = tf.combine_time(bars[-1]["frame"], 14, 57)
                self._buy(code, shares, order_time)

    @staticmethod
    def start_backtest(start: str, end: str = None):
        start = arrow.get(start).date()
        end = arrow.now() if end is None else arrow.get(end).date()

        os.environ[cfg4py.envar] = "PRODUCTION"
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""

        cfg4py.init(get_config_dir())

        s = BreakoutStrategy(url)
        asyncio.run(s.run(start, end))


if __name__ == "__main__":
    fire.Fire({"bt": BreakoutStrategy.start_backtest})
