import asyncio
import datetime
import json
import logging
import os
import uuid
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Dict

import cfg4py
from coretypes import Frame
from omicron import cache, tf
from traderclient import OrderType, TradeClient

from alpha.core.errors import TaskIsRunningError

logger = logging.getLogger(__name__)


class BaseStrategy(object, metaclass=ABCMeta):
    """所有Strategy的基类。

    本基类实现了进行一次回测时所需要的通用方法，比如，更新回测状态，获取回测结果等，记录账户的一些基本信息，比如，账户可用资金，账户持仓等。

    子类需要实现backtest方法，以实现具体的回测。

    本模块依赖于zillionare-backtest和zillionare-trader-client库。

    回测状态:
    每次回测的进度数据都按list存储，其队列名为`backtest_progress:{strategy-name}:{hash[-6:]}`。另外用`backtest_progress:{strategy-name}:{hash[-6:]}:metrics`来存储回测的评估结果。

    Args:

    Raises:
        TaskIsRunningError: _description_
        NotImplementedError: _description_
    """

    name = "base-strategy"
    desc = "Base Strategy Class"

    # params needed by backtest, each item is a tuple of (name, desc, default)
    backtest_params = []

    def __init__(
        self,
        is_backtest: bool = True,
        broker: TradeClient = None,
        mdd: float = 0.1,
        sl: float = 0.05,
        capital: int = 1_000_000,
    ):
        """

        Args:
            is_backtest : 是回测模式还是实盘交易模式
            broker : 交易代理。如果为回测模式，则此值被忽略，策略将构建一个新的代理。
            mdd : 止损前允许的最大回撤
            sl : 止损前允许的最大亏损.
            capital : 本金，如果是实盘交易模式，传入值将被忽略.
        """
        self._running_backtest = None
        self._is_backtest = is_backtest
        self._last_progress_dt = None

        self.positions = {}
        self.balance = None
        self.cash = None

        # 用于策略止损的参数
        self.thresholds = {"mdd": mdd, "sl": sl}

        if is_backtest:
            token = uuid.uuid4().hex
            account = f"{self.name}-{token[-4:]}"
            cfg = cfg4py.get_instance()
            url = cfg.backtest.url

            self.cash = capital
            self.broker = TradeClient(
                url, account, token, is_backtest=True, captipal=self.cash
            )

            self._bought = defaultdict(list)
            self._sold = defaultdict(list)
        else:
            self.broker = broker
            cash = self.broker.available_money()
            if cash is None:
                raise ValueError("Failed to get available money from server")

            self.cash = cash

    async def run(self, start: datetime.date, end: datetime.date, *args, **kwargs):
        """运行策略回测"""
        if self._running_backtest is not None:
            raise TaskIsRunningError(
                f"A backtest task of {self.name} is running: {self._running_backtest}"
            )

        try:
            self._running_backtest = f"{self.name}:{uuid.uuid4().hex[-6:]}"
            metrics = f"backtest_progress:{self._running_backtest}:metrics"
            await cache.sys.hmset(
                metrics,
                "start",
                start.isoformat(),
                "end",
                end.isoformat(),
                "status",
                "launching",
            )
            await self.backtest(start, end, *args, **kwargs)
            # await cache.sys.hset(metrics, "status", "running")

            # 回测已经结束，获取回测评估分析
            metrics = self.broker.metrics(start, end)
            await cache.sys.hset(metrics, "status", "done")
            self._running_backtest = None
        except Exception:
            await cache.sys.hset(metrics, "status", "failed")
            self._running_backtest = None
            raise

    async def buy(
        self,
        code: str,
        price: float,
        percent: float,
        dt: Frame,
        market_price: bool = False,
    ):
        """买入股票

        Args:
            code: 股票代码
            price: 买入价格
            volume: 买入数量
            dt: 买入时间
            market_price: 是否使用市价买入。此时price仅用来计算买入手数。
        """
        if type(dt) is not datetime.datetime:
            dt = tf.combine_time(dt, 9, 31)

        volume = (self.cash * percent / price) // 100
        if market_price:
            result = self.broker.market_buy(
                code, volume, OrderType.MARKET, order_time=dt.isoformat()
            )
        else:
            result = self.broker.buy(code, price, volume, order_time=dt.isoformat())

        # 更新持仓、可用资金
        self.update_portifolio()

        if result is not None:
            for trade in result["data"]:
                self._bought[dt.date()].append(
                    (trade["code"], trade["price"], trade["filled"])
                )

    async def sell(
        self,
        code: str,
        price: float,
        volume: int,
        dt: Frame,
        market_price: bool = False,
    ):
        """卖出持仓股票

        Args:
            code : _description_
            price : _description_
            volume : _description_
            dt : _description_
            market_price : _description_.
        """
        if type(dt) is not datetime.datetime:
            dt = tf.combine_time(dt, 9, 31)

        if market_price:
            result = self.broker.market_sell(
                code, volume, OrderType.MARKET, order_time=dt.isoformat()
            )
        else:
            result = self.broker.sell(code, price, volume, order_time=dt.isoformat())

        # 更新持仓、可用资金。如果是回测，更新回测进度
        self.update_portifolio()

        if result is not None:
            for trade in result["data"]:
                self._sold[dt.date()].append(
                    (trade["code"], trade["price"], trade["filled"])
                )

    async def update_backtest_progress(self, dt: datetime.date):
        """更新回测进度"""
        if dt == self._last_progress_dt:
            return

        if self._running_backtest is None:
            raise ValueError("No backtest is running")

        queue = f"backtest_progress:{self._running_backtest}"

        await cache.sys.rpush(
            queue,
            json.dumps(
                {
                    "date": dt.isoformat(),
                    "cash": self.cash,
                    "pnl": self.balance["pnl"],
                    "assets": self.balance["total"],
                    "ppnl": self.balance["ppnl"],
                    "market_value": self.balance["market_value"],
                    "positions": self.positions,
                    "bought": self._bought[dt],
                    "sold": self._sold[dt],
                }
            ),
        )

        self._last_progress_dt = dt

    def update_portifolio(self):
        """更新当前可用现金和持仓"""
        self.balance = self.broker.balance()["data"]
        self.cash = self.balance["available"]

        positions = self.broker.positions()["data"]
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

    @abstractmethod
    def backtest(self, *args, **kwargs):
        """调用子类的回测函数以启动回测

        Args:
            window: 回测的天数
        """
        raise NotImplementedError
