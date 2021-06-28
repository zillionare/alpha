from alpha.backtesting.forward_array import ForwardArray
from alpha.backtesting.errors import OutOfMoneyError
import sys
from abc import ABCMeta, abstractmethod
from itertools import chain
from numbers import Number
from typing import Callable, Tuple, Union

import numpy as np
import logging
from numpy.typing import ArrayLike
import pandas as pd

from alpha.backtesting.broker import Broker
from alpha.backtesting.order import Order
from alpha.backtesting.position import Position
from alpha.backtesting.trade import Trade
from omicron.models.security import Security


logger = logging.getLogger()

def _as_str(value) -> str:
    if isinstance(value, (Number, str)):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return "df"
    name = str(getattr(value, "name", "") or "")
    if name in ("Open", "High", "Low", "Close", "Volume"):
        return name[:1]
    if callable(value):
        name = getattr(value, "__name__", value.__class__.__name__).replace(
            "<lambda>", "λ"
        )
    if len(name) > 10:
        name = name[:9] + "…"
    return name


class Strategy(metaclass=ABCMeta):
    """
    A trading strategy base class. Extend this class and override methods
    `backtesting.backtesting.Strategy.init` and
    `backtesting.backtesting.Strategy.next` to define
    your own strategy.
    """

    def __init__(self, broker: Broker, features: ForwardArray, **params):
        """init the strategy

        Args:
            broker (Broker): [description]
            features (ForwardArray): 
        """
        self._broker = broker
        self._params = self._check_params(params)
        self._features = features

    def __repr__(self):
        return "<Strategy " + str(self) + ">"

    def __str__(self):
        params = ",".join(
            f"{i[0]}={i[1]}"
            for i in zip(self._params.keys(), map(_as_str, self._params.values()))
        )
        if params:
            params = "(" + params + ")"
        return f"{self.__class__.__name__}{params}"

    def _check_params(self, params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(
                    f"Strategy '{self.__class__.__name__}' is missing parameter '{k}'."
                    "Strategy class should define parameters as class variables before they "
                    "can be optimized or run with."
                )
            setattr(self, k, v)
        return params

    @abstractmethod
    async def init(self):
        """
        Initialize the strategy.
        Override this method.
        Declare indicators (with `backtesting.backtesting.Strategy.I`).
        Precompute what needs to be precomputed or can be precomputed
        in a vectorized fashion before the strategy starts.

        If you extend composable strategies from `backtesting.lib`,
        make sure to call:

            super().init()
        """

    @abstractmethod
    async def next(self):
        """
        Main strategy runtime method, called as each new
        `backtesting.backtesting.Strategy.data`
        instance (row; full candlestick bar) becomes available.
        This is the main method where strategy decisions
        upon data precomputed in `backtesting.backtesting.Strategy.init`
        take place.

        If you extend composable strategies from `backtesting.lib`,
        make sure to call:

            super().next()
        """

    async def backtest(self):
        await self.init()
        while self._features.reveal():
            try:
                self._broker.next()
            except OutOfMoneyError:
                break

            await self.next()
        else:
            # close any remaining open trades so they produce some stats
            for trade in self._broker.trades:
                trade.close()

            # Re-run broker one last time to handle orders placed in the last strategy
            # iteration. Use the same OHLC values as in the last broker iteration.
            if self._features._cur < self._features.size:
                try:
                    self._broker.next()
                except OutOfMoneyError:
                    pass

        self._results = self._compute_stats()

        return self._results
    class __FULL_EQUITY(float):
        def __repr__(self):
            return ".9999"

    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

    async def buy(
        self,
        *,
        size: float = _FULL_EQUITY,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
    ):
        """
        Place a new long order and emit buy emit

        For explanation of parameters, see `Order` and its properties.

        See also `Strategy.sell()`.
        """
        assert (
            0 < size < 1 or round(size) == size
        ), "size must be a positive fraction of equity, or a positive whole number of units"
        order = self._broker.new_order(size, limit, stop, sl, tp)

        # todo: emit signal
        return order

    async def sell(
        self,
        *,
        size: float = _FULL_EQUITY,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
    ):
        """
        Place a new short order and emit sell event.

        For explanation of parameters, see `Order` and its properties.

        See also `Strategy.buy()`.
        """
        assert (
            0 < size < 1 or round(size) == size
        ), "size must be a positive fraction of equity, or a positive whole number of units"
        order = self._broker.new_order(-size, limit, stop, sl, tp)

        # todo: emit signal
        return order

    async def close_position(self):
        """close position currently hold and emit close_position event"""
        self.position.close()

        # todo: emit the signal

    @property
    def equity(self) -> float:
        """Current account equity (cash plus assets)."""
        return self._broker.equity

    @property
    def features(self) -> Security:
        return self._features

    @property
    def position(self) -> Position:
        """Instance of `backtesting.backtesting.Position`."""
        return self._broker.position

    @property
    def orders(self) -> Tuple[Order, ...]:
        """List of orders (see `Order`) waiting for execution."""
        return self._broker.orders

    @property
    def trades(self) -> Tuple[Trade, ...]:
        """List of active trades (see `Trade`)."""
        return tuple(self._broker.trades)

    @property
    def closed_trades(self) -> Tuple[Trade, ...]:
        """List of settled trades (see `Trade`)."""
        return tuple(self._broker.closed_trades)

    def _compute_stats(self):
        broker = self._broker
        index = self._features['frame']

        equity = pd.Series(broker._equity).bfill().fillna(broker._cash).values
        dd = 1 - equity / np.maximum.accumulate(equity)
        dd_dur, dd_peaks = self._compute_drawdown_duration_peaks(pd.Series(dd, index=index))

        equity_df = pd.DataFrame(
            {"Equity": equity, "DrawdownPct": dd, "DrawdownDuration": dd_dur},
            index=index,
        )

        trades = broker.closed_trades
        trades_df = pd.DataFrame(
            {
                "Size": [t.size for t in trades],
                "EntryBar": [t.entry_bar for t in trades],
                "ExitBar": [t.exit_bar for t in trades],
                "EntryPrice": [t.entry_price for t in trades],
                "ExitPrice": [t.exit_price for t in trades],
                "PnL": [t.pl for t in trades],
                "ReturnPct": [t.pl_pct for t in trades],
                "EntryTime": [t.entry_time for t in trades],
                "ExitTime": [t.exit_time for t in trades],
            }
        )
        trades_df["Duration"] = trades_df["ExitTime"] - trades_df["EntryTime"]

        pl = trades_df["PnL"]
        returns = trades_df["ReturnPct"]
        durations = trades_df["Duration"]

        def _round_timedelta(value, _period=self.period(index)):
            if not isinstance(value, pd.Timedelta):
                return value
            resolution = (
                getattr(_period, "resolution_string", None) or _period.resolution
            )
            return value.ceil(resolution)

        s = pd.Series(dtype=object)
        s.loc["Start"] = index[0]
        s.loc["End"] = index[-1]
        s.loc["Duration"] = s.End - s.Start

        have_position = np.repeat(0, len(index))
        for t in trades:
            have_position[t.entry_bar : t.exit_bar + 1] = 1  # type: ignore

        s.loc["Exposure Time [%]"] = (
            have_position.mean() * 100
        )  # In "n bars" time, not index time
        s.loc["Equity Final [$]"] = equity[-1]
        s.loc["Equity Peak [$]"] = equity.max()
        s.loc["Return [%]"] = (equity[-1] - equity[0]) / equity[0] * 100
        c = self.features["close"]
        s.loc["Buy & Hold Return [%]"] = (c[-1] - c[0]) / c[0] * 100  # long-only return

        def geometric_mean(returns):
            returns = returns.fillna(0) + 1
            return (
                0
                if np.any(returns <= 0)
                else np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1
            )

        day_returns = gmean_day_return = np.array(np.nan)
        annual_trading_days = np.nan
        if isinstance(index, pd.DatetimeIndex):
            day_returns = equity_df["Equity"].resample("D").last().dropna().pct_change()
            gmean_day_return = geometric_mean(day_returns)
            annual_trading_days = float(
                365
                if index.dayofweek.to_series().between(5, 6).mean() > 2 / 7 * 0.6
                else 252
            )

        # Annualized return and risk metrics are computed based on the (mostly correct)
        # assumption that the returns are compounded. See: https://dx.doi.org/10.2139/ssrn.3054517
        # Our annualized return matches `empyrical.annual_return(day_returns)` whereas
        # our risk doesn't; they use the simpler approach below.
        annualized_return = (1 + gmean_day_return) ** annual_trading_days - 1
        s.loc["Return (Ann.) [%]"] = annualized_return * 100
        s.loc["Volatility (Ann.) [%]"] = (
            np.sqrt(
                (
                    day_returns.var(ddof=int(bool(day_returns.shape)))
                    + (1 + gmean_day_return) ** 2
                )
                ** annual_trading_days
                - (1 + gmean_day_return) ** (2 * annual_trading_days)
            )
            * 100
        )  # noqa: E501
        # s.loc['Return (Ann.) [%]'] = gmean_day_return * annual_trading_days * 100
        # s.loc['Risk (Ann.) [%]'] = day_returns.std(ddof=1) * np.sqrt(annual_trading_days) * 100

        # Our Sharpe mismatches `empyrical.sharpe_ratio()` because they use arithmetic mean return
        # and simple standard deviation
        s.loc["Sharpe Ratio"] = np.clip(
            s.loc["Return (Ann.) [%]"] / (s.loc["Volatility (Ann.) [%]"] or np.nan),
            0,
            np.inf,
        )  # noqa: E501
        # Our Sortino mismatches `empyrical.sortino_ratio()` because they use arithmetic mean return
        s.loc["Sortino Ratio"] = np.clip(
            annualized_return
            / (
                np.sqrt(np.mean(day_returns.clip(-np.inf, 0) ** 2))
                * np.sqrt(annual_trading_days)
            ),
            0,
            np.inf,
        )  # noqa: E501
        max_dd = -np.nan_to_num(dd.max())
        s.loc["Calmar Ratio"] = np.clip(
            annualized_return / (-max_dd or np.nan), 0, np.inf
        )
        s.loc["Max. Drawdown [%]"] = max_dd * 100
        s.loc["Avg. Drawdown [%]"] = -dd_peaks.mean() * 100
        s.loc["Max. Drawdown Duration"] = _round_timedelta(dd_dur.max())
        s.loc["Avg. Drawdown Duration"] = _round_timedelta(dd_dur.mean())
        s.loc["# Trades"] = n_trades = len(trades)
        s.loc["Win Rate [%]"] = (
            np.nan if not n_trades else (pl > 0).sum() / n_trades * 100
        )  # noqa: E501
        s.loc["Best Trade [%]"] = returns.max() * 100
        s.loc["Worst Trade [%]"] = returns.min() * 100
        mean_return = geometric_mean(returns)
        s.loc["Avg. Trade [%]"] = mean_return * 100
        s.loc["Max. Trade Duration"] = _round_timedelta(durations.max())
        s.loc["Avg. Trade Duration"] = _round_timedelta(durations.mean())
        s.loc["Profit Factor"] = returns[returns > 0].sum() / (
            abs(returns[returns < 0].sum()) or np.nan
        )  # noqa: E501
        s.loc["Expectancy [%]"] = returns.mean() * 100
        s.loc["SQN"] = np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)

        s.loc["_strategy"] = self
        s.loc["_equity_curve"] = equity_df
        s.loc["_trades"] = trades_df

        return s

    def _compute_drawdown_duration_peaks(self, dd:pd.Series):
        iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
        iloc = pd.Series(iloc, index=dd.index[iloc])
        df = iloc.to_frame("iloc").assign(prev=iloc.shift())
        df = df[df["iloc"] > df["prev"] + 1].astype(int)
        # If no drawdown since no trade, avoid below for pandas sake and return nan series
        if not len(df):
            return (dd.replace(0, np.nan),) * 2
        df["duration"] = df["iloc"].map(dd.index.__getitem__) - df["prev"].map(
            dd.index.__getitem__
        )
        df["peak_dd"] = df.apply(
            lambda row: dd.iloc[row["prev"] : row["iloc"] + 1].max(), axis=1
        )
        df = df.reindex(dd.index)
        return df["duration"], df["peak_dd"]

    def period(self, frame: np.array) -> Union[pd.Timedelta, Number]:
        """Return data index period as pd.Timedelta

        Copied from backtesting.util._data_period

        Args:
            frame (np.array): [description]

        Returns:
            Union[pd.Timedelta, Number]: [description]
        """
        values = pd.Series(frame[-100:])
        return values.diff().dropna().median()
