"""
Core framework data structures.
Objects from this module can also be imported from the top-level
module directly
"""
import logging
import multiprocessing as mp
import os
import warnings
from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache, partial
from itertools import chain, compress, product, repeat
from math import copysign
from numbers import Number
from typing import Callable, Dict, List, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd

from alpha.backtesting.broker import Broker
from alpha.backtesting.errors import OutOfMoneyError
from alpha.backtesting.indicator import Indicator
from alpha.backtesting.strategy import Strategy
from alpha.plotting.backtest_result import plot
from omicron.models.security import Security

logger = logging.getLogger(__name__)


def period(frame: np.array) -> Union[pd.Timedelta, Number]:
    """Return data index period as pd.Timedelta

    Copied from backtesting.util._data_period

    Args:
        frame (np.array): [description]

    Returns:
        Union[pd.Timedelta, Number]: [description]
    """
    values = pd.Series(frame[-100:])
    return values.diff().dropna().median()


class Backtest:
    """
    Backtest a particular (parameterized) strategy
    on particular data.

    Upon initialization, call method
    `backtesting.backtesting.Backtest.run` to run a backtest
    instance, or `backtesting.backtesting.Backtest.optimize` to
    optimize it.
    """

    def __init__(
        self,
        sec: Security,
        strategy: Type[Strategy],
        *,
        cash: float = 10_000,  # PEP515, since 3.6
        commission: float = 0.0,
        margin: float = 1.0,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=False,
    ):
        self._sec = sec

        self._broker = Broker(
            data=sec,
            cash=cash,
            commission=commission,
            margin=margin,
            trade_on_close=trade_on_close,
            hedging=hedging,
            exclusive_orders=exclusive_orders,
            index=sec.frame,
        )

        self._strategy = strategy(self._broker, data=sec)

        self._results = None

    async def run(self, **kwargs) -> pd.Series:
        """
        Run the backtest. Returns `pd.Series` with results and statistics.

        Keyword arguments are interpreted as strategy parameters.

            >>> Backtest(GOOG, SmaCross).run()
            Start                     2004-08-19 00:00:00
            End                       2013-03-01 00:00:00
            Duration                   3116 days 00:00:00
            Exposure Time [%]                     93.9944
            Equity Final [$]                      51959.9
            Equity Peak [$]                       75787.4
            Return [%]                            419.599
            Buy & Hold Return [%]                 703.458
            Return (Ann.) [%]                      21.328
            Volatility (Ann.) [%]                 36.5383
            Sharpe Ratio                         0.583718
            Sortino Ratio                         1.09239
            Calmar Ratio                         0.444518
            Max. Drawdown [%]                    -47.9801
            Avg. Drawdown [%]                    -5.92585
            Max. Drawdown Duration      584 days 00:00:00
            Avg. Drawdown Duration       41 days 00:00:00
            # Trades                                   65
            Win Rate [%]                          46.1538
            Best Trade [%]                         53.596
            Worst Trade [%]                      -18.3989
            Avg. Trade [%]                        2.35371
            Max. Trade Duration         183 days 00:00:00
            Avg. Trade Duration          46 days 00:00:00
            Profit Factor                         2.08802
            Expectancy [%]                        8.79171
            SQN                                  0.916893
            _strategy                            SmaCross
            _equity_curve                           Eq...
            _trades                       Size  EntryB...
            dtype: object
        """
        broker: Broker = self._broker
        strategy: Strategy = self._strategy
        await strategy.init()

        # Indicators used in Strategy.next()
        indicator_attrs = {
            attr: indicator
            for attr, indicator in strategy.__dict__.items()
            if isinstance(indicator, Indicator)
        }.items()

        # Skip first few candles where indicators are still "warming up"
        # +1 to have at least two entries available
        start = 1 + max(
            (
                np.isnan(indicator.astype(float)).argmin(axis=-1).max()
                for _, indicator in indicator_attrs
            ),
            default=0,
        )

        # Disable "invalid value encountered in ..." warnings. Comparison
        # np.nan >= 3 is not invalid; it's False.
        with np.errstate(invalid="ignore"):

            for i in range(start, len(self._sec)):
                # Prepare data and indicators for `next` call
                self._sec.set_size(i + 1)
                for attr, indicator in indicator_attrs:
                    # Slice indicator on the last dimension (case of 2d indicator)
                    setattr(strategy, attr, indicator[..., : i + 1])

                # Handle orders processing and broker stuff
                try:
                    broker.next()
                except OutOfMoneyError:
                    break

                # Next tick, a moment before bar close
                await strategy.next()
            else:
                # Close any remaining open trades so they produce some stats
                for trade in broker.trades:
                    trade.close()

                # Re-run broker one last time to handle orders placed in the last strategy
                # iteration. Use the same OHLC values as in the last broker iteration.
                if start < len(self._sec):
                    try:
                        broker.next()
                    except OutOfMoneyError:
                        pass

            # Set data back to full length
            # for future `indicator._opts['data'].index` calls to work
            self._sec.reset_size()

            self._results = self._compute_stats(broker, strategy)
        return self._results

    def optimize(
        self,
        *,
        maximize: Union[str, Callable[[pd.Series], float]] = "SQN",
        method: str = "grid",
        max_tries: Union[int, float] = None,
        constraint: Callable[[dict], bool] = None,
        return_heatmap: bool = False,
        return_optimization: bool = False,
        random_state: int = None,
        **kwargs,
    ) -> Union[
        pd.Series, Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, dict]
    ]:
        """
        Optimize strategy parameters to an optimal combination.
        Returns result `pd.Series` of the best run.

        `maximize` is a string key from the
        `backtesting.backtesting.Backtest.run`-returned results series,
        or a function that accepts this series object and returns a number;
        the higher the better. By default, the method maximizes
        Van Tharp's [System Quality Number](https://google.com/search?q=System+Quality+Number).

        `method` is the optimization method. Currently two methods are supported:

        * `"grid"` which does an exhaustive (or randomized) search over the
          cartesian product of parameter combinations, and
        * `"skopt"` which finds close-to-optimal strategy parameters using
          [model-based optimization], making at most `max_tries` evaluations.

        [model-based optimization]: \
            https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html

        `max_tries` is the maximal number of strategy runs to perform.
        If `method="grid"`, this results in randomized grid search.
        If `max_tries` is a floating value between (0, 1], this sets the
        number of runs to approximately that fraction of full grid space.
        Alternatively, if integer, it denotes the absolute maximum number
        of evaluations. If unspecified (default), grid search is exhaustive,
        whereas for `method="skopt"`, `max_tries` is set to 200.

        `constraint` is a function that accepts a dict-like object of
        parameters (with values) and returns `True` when the combination
        is admissible to test with. By default, any parameters combination
        is considered admissible.

        If `return_heatmap` is `True`, besides returning the result
        series, an additional `pd.Series` is returned with a multiindex
        of all admissible parameter combinations, which can be further
        inspected or projected onto 2D to plot a heatmap
        (see `backtesting.lib.plot_heatmaps()`).

        If `return_optimization` is True and `method = 'skopt'`,
        in addition to result series (and maybe heatmap), return raw
        [`scipy.optimize.OptimizeResult`][OptimizeResult] for further
        inspection, e.g. with [scikit-optimize]\
        [plotting tools].

        [OptimizeResult]: \
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
        [scikit-optimize]: https://scikit-optimize.github.io
        [plotting tools]: https://scikit-optimize.github.io/stable/modules/plots.html

        If you want reproducible optimization results, set `random_state`
        to a fixed integer or a `numpy.random.RandomState` object.

        Additional keyword arguments represent strategy arguments with
        list-like collections of possible values. For example, the following
        code finds and returns the "best" of the 7 admissible (of the
        9 possible) parameter combinations:

            backtest.optimize(sma1=[5, 10, 15], sma2=[10, 20, 40],
                              constraint=lambda p: p.sma1 < p.sma2)

        .. TODO::
            Improve multiprocessing/parallel execution on Windos with start method 'spawn'.
        """
        if not kwargs:
            raise ValueError("Need some strategy parameters to optimize")

        maximize_key = None
        if isinstance(maximize, str):
            maximize_key = str(maximize)
            stats = self._results if self._results is not None else self.run()
            if maximize not in stats:
                raise ValueError(
                    "`maximize`, if str, must match a key in pd.Series "
                    "result of backtest.run()"
                )

            def maximize(stats: pd.Series, _key=maximize):
                return stats[_key]

        elif not callable(maximize):
            raise TypeError(
                "`maximize` must be str (a field of backtest.run() result "
                "Series) or a function that accepts result Series "
                "and returns a number; the higher the better"
            )

        have_constraint = bool(constraint)
        if constraint is None:

            def constraint(_):
                return True

        elif not callable(constraint):
            raise TypeError(
                "`constraint` must be a function that accepts a dict "
                "of strategy parameters and returns a bool whether "
                "the combination of parameters is admissible or not"
            )

        if return_optimization and method != "skopt":
            raise ValueError("return_optimization=True only valid if method='skopt'")

        def _tuple(x):
            return x if isinstance(x, Sequence) and not isinstance(x, str) else (x,)

        for k, v in kwargs.items():
            if len(_tuple(v)) == 0:
                raise ValueError(
                    f"Optimization variable '{k}' is passed no "
                    f"optimization values: {k}={v}"
                )

        class AttrDict(dict):
            def __getattr__(self, item):
                return self[item]

        def _grid_size():
            size = np.prod([len(_tuple(v)) for v in kwargs.values()])
            if size < 10_000 and have_constraint:
                size = sum(
                    1
                    for p in product(
                        *(zip(repeat(k), _tuple(v)) for k, v in kwargs.items())
                    )
                    if constraint(AttrDict(p))
                )
            return size

        def _optimize_grid() -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
            rand = np.random.RandomState(random_state).random
            grid_frac = (
                1
                if max_tries is None
                else max_tries
                if 0 < max_tries <= 1
                else max_tries / _grid_size()
            )
            param_combos = [
                dict(params)  # back to dict so it pickles
                for params in (
                    AttrDict(params)
                    for params in product(
                        *(zip(repeat(k), _tuple(v)) for k, v in kwargs.items())
                    )
                )
                if constraint(params) and rand() <= grid_frac  # type: ignore
            ]
            if not param_combos:
                raise ValueError("No admissible parameter combinations to test")

            if len(param_combos) > 300:
                warnings.warn(
                    f"Searching for best of {len(param_combos)} configurations.",
                    stacklevel=2,
                )

            heatmap = pd.Series(
                np.nan,
                name=maximize_key,
                index=pd.MultiIndex.from_tuples(
                    [p.values() for p in param_combos],
                    names=next(iter(param_combos)).keys(),
                ),
            )

            def _batch(seq):
                n = np.clip(int(len(seq) // (os.cpu_count() or 1)), 1, 300)
                for i in range(0, len(seq), n):
                    yield seq[i : i + n]

            # Save necessary objects into "global" state; pass into concurrent executor
            # (and thus pickle) nothing but two numbers; receive nothing but numbers.
            # With start method "fork", children processes will inherit parent address space
            # in a copy-on-write manner, achieving better performance/RAM benefit.
            backtest_uuid = np.random.random()
            param_batches = list(_batch(param_combos))
            Backtest._mp_backtests[backtest_uuid] = (self, param_batches, maximize)  # type: ignore
            try:
                with ProcessPoolExecutor() as executor:
                    futures = [
                        executor.submit(Backtest._mp_task, backtest_uuid, i)
                        for i in range(len(param_batches))
                    ]
                    for future in as_completed(futures):
                        batch_index, values = future.result()
                        for value, params in zip(values, param_batches[batch_index]):
                            heatmap[tuple(params.values())] = value
            finally:
                del Backtest._mp_backtests[backtest_uuid]

            best_params = heatmap.idxmax()

            if pd.isnull(best_params):
                # No trade was made in any of the runs. Just make a random
                # run so we get some, if empty, results
                stats = self.run(**param_combos[0])
            else:
                stats = self.run(**dict(zip(heatmap.index.names, best_params)))

            if return_heatmap:
                return stats, heatmap
            return stats

        def _optimize_skopt() -> Union[
            pd.Series, Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, dict]
        ]:
            try:
                from skopt import forest_minimize
                from skopt.callbacks import DeltaXStopper
                from skopt.learning import ExtraTreesRegressor
                from skopt.space import Categorical, Integer, Real
                from skopt.utils import use_named_args
            except ImportError:
                raise ImportError(
                    "Need package 'scikit-optimize' for method='skopt'. "
                    "pip install scikit-optimize"
                )

            nonlocal max_tries
            max_tries = (
                200
                if max_tries is None
                else max(1, int(max_tries * _grid_size()))
                if 0 < max_tries <= 1
                else max_tries
            )

            dimensions = []
            for key, values in kwargs.items():
                values = np.asarray(values)
                if values.dtype.kind in "mM":  # timedelta, datetime64
                    # these dtypes are unsupported in skopt, so convert to raw int
                    # TODO: save dtype and convert back later
                    values = values.astype(int)

                if values.dtype.kind in "iumM":
                    dimensions.append(
                        Integer(low=values.min(), high=values.max(), name=key)
                    )
                elif values.dtype.kind == "f":
                    dimensions.append(
                        Real(low=values.min(), high=values.max(), name=key)
                    )
                else:
                    dimensions.append(
                        Categorical(values.tolist(), name=key, transform="onehot")
                    )

            # Avoid recomputing re-evaluations:
            # "The objective has been evaluated at this point before."
            # https://github.com/scikit-optimize/scikit-optimize/issues/302
            memoized_run = lru_cache()(lambda tup: self.run(**dict(tup)))

            # np.inf/np.nan breaks sklearn, np.finfo(float).max breaks skopt.plots.plot_objective
            INVALID = 1e300

            @use_named_args(dimensions=dimensions)
            def objective_function(**params):
                # Check constraints
                # TODO: Adjust after https://github.com/scikit-optimize/scikit-optimize/pull/971
                if not constraint(AttrDict(params)):
                    return INVALID
                res = memoized_run(tuple(params.items()))
                value = -maximize(res)
                if np.isnan(value):
                    return INVALID
                return value

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "The objective has been evaluated at this point before."
                )

                res = forest_minimize(
                    func=objective_function,
                    dimensions=dimensions,
                    n_calls=max_tries,
                    base_estimator=ExtraTreesRegressor(
                        n_estimators=20, min_samples_leaf=2
                    ),
                    acq_func="LCB",
                    kappa=3,
                    n_initial_points=min(max_tries, 20 + 3 * len(kwargs)),
                    initial_point_generator="lhs",  # 'sobel' requires n_initial_points ~ 2**N
                    callback=DeltaXStopper(9e-7),
                    random_state=random_state,
                )

            stats = self.run(**dict(zip(kwargs.keys(), res.x)))
            output = [stats]

            if return_heatmap:
                heatmap = pd.Series(
                    dict(zip(map(tuple, res.x_iters), -res.func_vals)),
                    name=maximize_key,
                )
                heatmap.index.names = kwargs.keys()
                heatmap = heatmap[heatmap != -INVALID]
                heatmap.sort_index(inplace=True)
                output.append(heatmap)

            if return_optimization:
                valid = res.func_vals != INVALID
                res.x_iters = list(compress(res.x_iters, valid))
                res.func_vals = res.func_vals[valid]
                output.append(res)

            return stats if len(output) == 1 else tuple(output)

        if method == "grid":
            output = _optimize_grid()
        elif method == "skopt":
            output = _optimize_skopt()
        else:
            raise ValueError(f"Method should be 'grid' or 'skopt', not {method!r}")
        return output

    @staticmethod
    def _mp_task(backtest_uuid, batch_index):
        bt, param_batches, maximize_func = Backtest._mp_backtests[backtest_uuid]
        return batch_index, [
            maximize_func(stats) if stats["# Trades"] else np.nan
            for stats in (bt.run(**params) for params in param_batches[batch_index])
        ]

    _mp_backtests: Dict[float, Tuple["Backtest", List, Callable]] = {}

    @staticmethod
    def _compute_drawdown_duration_peaks(dd: pd.Series):
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

    def _compute_stats(self, broker: Broker, strategy: Strategy) -> pd.Series:
        data = self._sec
        index = data.index

        equity = pd.Series(broker._equity).bfill().fillna(broker._cash).values
        dd = 1 - equity / np.maximum.accumulate(equity)
        dd_dur, dd_peaks = self._compute_drawdown_duration_peaks(
            pd.Series(dd, index=index)
        )

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

        def _round_timedelta(value, _period=period(index)):
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
        c = data.Close.values
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

        s.loc["_strategy"] = strategy
        s.loc["_equity_curve"] = equity_df
        s.loc["_trades"] = trades_df

        s = Backtest._Stats(s)
        return s

    class _Stats(pd.Series):
        def __repr__(self):
            # Prevent expansion due to _equity and _trades dfs
            with pd.option_context("max_colwidth", 20):
                return super().__repr__()

    def plot(
        self,
        *,
        results: pd.Series = None,
        filename=None,
        plot_width=None,
        plot_equity=True,
        plot_return=False,
        plot_pl=True,
        plot_volume=True,
        plot_drawdown=False,
        smooth_equity=False,
        relative_equity=True,
        superimpose: Union[bool, str] = True,
        resample=True,
        reverse_indicators=False,
        show_legend=True,
        open_browser=True,
    ):
        """
        Plot the progression of the last backtest run.

        If `results` is provided, it should be a particular result
        `pd.Series` such as returned by
        `backtesting.backtesting.Backtest.run` or
        `backtesting.backtesting.Backtest.optimize`, otherwise the last
        run's results are used.

        `filename` is the path to save the interactive HTML plot to.
        By default, a strategy/parameter-dependent file is created in the
        current working directory.

        `plot_width` is the width of the plot in pixels. If None (default),
        the plot is made to span 100% of browser width. The height is
        currently non-adjustable.

        If `plot_equity` is `True`, the resulting plot will contain
        an equity (initial cash plus assets) graph section. This is the same
        as `plot_return` plus initial 100%.

        If `plot_return` is `True`, the resulting plot will contain
        a cumulative return graph section. This is the same
        as `plot_equity` minus initial 100%.

        If `plot_pl` is `True`, the resulting plot will contain
        a profit/loss (P/L) indicator section.

        If `plot_volume` is `True`, the resulting plot will contain
        a trade volume section.

        If `plot_drawdown` is `True`, the resulting plot will contain
        a separate drawdown graph section.

        If `smooth_equity` is `True`, the equity graph will be
        interpolated between fixed points at trade closing times,
        unaffected by any interim asset volatility.

        If `relative_equity` is `True`, scale and label equity graph axis
        with return percent, not absolute cash-equivalent values.

        If `superimpose` is `True`, superimpose larger-timeframe candlesticks
        over the original candlestick chart. Default downsampling rule is:
        monthly for daily data, daily for hourly data, hourly for minute data,
        and minute for (sub-)second data.
        `superimpose` can also be a valid [Pandas offset string],
        such as `'5T'` or `'5min'`, in which case this frequency will be
        used to superimpose.
        Note, this only works for data with a datetime index.

        If `resample` is `True`, the OHLC data is resampled in a way that
        makes the upper number of candles for Bokeh to plot limited to 10_000.
        This may, in situations of overabundant data,
        improve plot's interactive performance and avoid browser's
        `Javascript Error: Maximum call stack size exceeded` or similar.
        Equity & dropdown curves and individual trades data is,
        likewise, [reasonably _aggregated_][TRADES_AGG].
        `resample` can also be a [Pandas offset string],
        such as `'5T'` or `'5min'`, in which case this frequency will be
        used to resample, overriding above numeric limitation.
        Note, all this only works for data with a datetime index.

        If `reverse_indicators` is `True`, the indicators below the OHLC chart
        are plotted in reverse order of declaration.

        [Pandas offset string]: \
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

        [TRADES_AGG]: lib.html#backtesting.lib.TRADES_AGG

        If `show_legend` is `True`, the resulting plot graphs will contain
        labeled legends.

        If `open_browser` is `True`, the resulting `filename` will be
        opened in the default web browser.
        """
        if results is None:
            if self._results is None:
                raise RuntimeError("First issue `backtest.run()` to obtain results.")
            results = self._results

        plot(
            results=results,
            df=self._sec,
            indicators=results._strategy._indicators,
            filename=filename,
            plot_width=plot_width,
            plot_equity=plot_equity,
            plot_return=plot_return,
            plot_pl=plot_pl,
            plot_volume=plot_volume,
            plot_drawdown=plot_drawdown,
            smooth_equity=smooth_equity,
            relative_equity=relative_equity,
            superimpose=superimpose,
            resample=resample,
            reverse_indicators=reverse_indicators,
            show_legend=show_legend,
            open_browser=open_browser,
        )
