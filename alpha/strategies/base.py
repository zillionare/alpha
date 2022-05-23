import asyncio
import datetime
import json
import logging
from types import FrameType
import uuid
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Optional

import cfg4py
from coretypes import Frame
from h2o_wave.core import Expando, copy_expando
from omicron import cache, tf
from pyemit import emit
from traderclient import OrderType, TradeClient

from alpha.core.const import E_BACKTEST, E_STRATEGY
from alpha.core.errors import TaskIsRunningError

logger = logging.getLogger(__name__)

# Todo: 这个类将回测数据存入了redis，是一种求快的做法。实际上回测结果数据查询频次很低，使用者也往往是人类，所以可以存入数据库，而不是缓存。
# Todo: 当前考虑回测更多一些，后面还需要把回测和实盘结合起来


class BaseStrategy(object, metaclass=ABCMeta):
    """所有Strategy的基类。

    本基类实现了以下功能：
    1. 所有继承本基类的子类策略，都可以通过[alpha.strategies.get_all_strategies][] 获取到所有策略的列表，包括name和description。这些信息可用于在控制台上显示策略的列表。
    2. 策略实现`buy`， `sell`功能，调用结果会通过事件通知发送出来，以便回测框架、控制台来刷新和更新进度。
    3. 在实盘和回测之间无缝切换。

    本模块依赖于zillionare-backtest和zillionare-trader-client库。

    Args:

    Raises:
        TaskIsRunningError: _description_
        NotImplementedError: _description_
    """
    name = "base-strategy"
    alias = "base for all strategies"
    desc = "Base Strategy Class"
    version = "NA"

    def __init__(self, broker: TradeClient = None, mdd: float = 0.1, sl: float = 0.05):
        """

        Args:
            broker : 交易代理
            mdd : 止损前允许的最大回撤
            sl : 止损前允许的最大亏损
        """
        # 当前持仓 code -> {security: , shares: , sellable: sl: }
        self.positions = {}
        self.balance = None
        self.cash = None

        # 用于策略止损的参数
        self.thresholds = {"mdd": mdd, "sl": sl}

        if broker:
            self.broker = broker
            cash = self.broker.available_money()
            if cash is None:
                raise ValueError("Failed to get available money from server")

            self.cash = cash

        self._bt = None

    async def notify(self, msg: dict):
        """通知事件。

        Args:
            msg: dict

        Returns:

        """
        if self._bt is not None:
            msg.update(
                {
                    "account": self._bt._account,
                    "token": self._bt._token,
                }
            )
            await emit.emit(E_BACKTEST, msg)
        else:
            await emit.emit(E_STRATEGY, msg)

    async def update_progress(self, current_frame: Frame, frame_type: FrameType):
        """更新回测进度

        Args:
            frame : 最新的回测时间
        """
        if self._bt is None:
            logger.warning("Backtest is not running, can't update progress")
            return
        
        last_frame = tf.shift(current_frame, -1, frame_type)
        if self._bt._last_frame is None:
            self._bt._last_frame = last_frame
        else:
            msg = f"frame rewinded: {self._bt._last_frame} -> {last_frame}"
            assert last_frame >= self._bt._last_frame, msg
            self._bt._last_frame = last_frame

        response = self._bt._broker.balance(last_frame)
        balance = response.get("data")

        await self.notify({
            "event": "progress",
            "frame": last_frame,
            "balance": balance,
        })
        
    async def start_backtest(
        self,
        start: datetime.date,
        end: datetime.date,
        principal: float = 1_000_000,
        params: dict = None,
    ):
        """运行策略回测"""
        if self._bt is not None:
            raise TaskIsRunningError(
                f"A backtest task of {self.name} is running: {self._bt._token}"
            )

        self._bt = Expando(params)

        token = uuid.uuid4().hex
        account = f"{self.name}-{self.version}-{token[-4:]}"
        cfg = cfg4py.get_instance()
        url = cfg.backtest.url
        self._bt._broker = TradeClient(
            url,
            account,
            token,
            is_backtest=True,
            capital=principal,
            commission=cfg.backtest.commission,
            start=start,
            end=end,
        )

        self._bt._account = account
        self._bt._token = token
        self._bt._positions = {}

        try:
            await self.notify(
                msg={
                    "event": "started",
                    "principal": principal,
                    "commission": cfg.backtest.commission,
                    "start": start,
                    "end": end,
                }
            )

            await self.backtest(start, end)

            # 回测已经结束，获取回测评估分析
            metrics = self._bt._broker.metrics(start, end)
            await self.notify(
                {"event": "finished", "metrics": metrics, "start": start, "end": end}
            )
        except Exception as e:
            logger.exception(e)
            await self.notify(
                {
                    "strategy": self.name,
                    "event": "failed",
                    "account": self._bt._account,
                    "token": self._bt._token,
                    "msg": str(e),
                }
            )
            raise
        finally:
            self._bt = None

    async def buy(
        self,
        code: str,
        shares: float,
        order_time: Optional[Frame] = None,
        price: Optional[float] = None,
    ):
        """买入股票

        Args:
            code: 股票代码
            shares: 买入数量。如果在(0, 1]之间，则认为是按可用资金比率买入；否则认为是买入股数，此时必须为100的倍数
            price: 买入价格,如果为None，则以市价买入
            order_time: 下单时间,仅在回测时需要，实盘时，即使传入也会被忽略
        """
        if self._bt is not None:
            broker = self._bt._broker
            if type(order_time) == datetime.date:
                order_time = tf.combine_time(order_time, 14, 56)

            cash = self._bt._cash
        else:
            broker = self.broker
            cash = self.cash

        if price is not None and 0 < shares <= 1:
            volume = 100 * ((cash * shares / price) // 100)
        else:
            assert shares > 100
            volume = shares

        logger.info("buy: %s %s %s", code, volume, order_time)
        if price is None:
            response = broker.market_buy(code, volume, order_time=order_time)

        # 更新持仓、可用资金
        self.update_portifolio()

    async def sell(
        self,
        code: str,
        shares: float,
        order_time: Optional[Frame] = None,
        price: Optional[float] = None,
    ):
        """卖出持仓股票

        Args:
            code : 卖出的证券代码
            shares : 如果在(0, 1]之间，则为卖出持仓的比例，或者股数。
            order_time : 委卖时间，在实盘时不必要传入
            price : 卖出价。如果为None，则为市价卖出
        """
        if self._bt is not None:
            broker = self._bt._broker
            if type(order_time) == datetime.date:
                order_time = tf.combine_time(order_time, 14, 56)
            positions = self._bt._positions
        else:
            broker = self.broker
            positions = self.positions

        if 0 < shares <= 1:
            volume = positions.get(code).get("sellable", 0) * shares
        else:
            volume = shares

        sellable = positions.get(code, {}).get("sellable", 0)
        if sellable == 0:
            logger.warning("%s has no sellable shares", code)
            return

        logger.info("sell: %s %s %s", code, volume, order_time)

        if price is None:
            result = broker.market_sell(code, volume, order_time=order_time.isoformat())
        else:
            result = broker.sell(code, price, volume, order_time=order_time.isoformat())

        # 更新持仓、可用资金。如果是回测，更新回测进度
        self.update_portifolio()

    def update_portifolio(self):
        """更新当前可用现金和持仓"""
        if self._bt is not None:
            broker = self._bt._broker
            self._bt._positions = broker.positions()["data"]
            balance = broker.balance()["data"]
            self._bt._cash = balance["available"]

        else:
            broker = self.broker
            self.positions = broker.positions()["data"]
            balance = broker.balance()["data"]
            self.cash = balance["available"]

    @abstractmethod
    async def backtest(self, start: datetime.date, end: datetime.date):
        """调用子类的回测函数以启动回测

        Args:
            start: 回测起始时间
            end: 回测结束时间
        """
        raise NotImplementedError
