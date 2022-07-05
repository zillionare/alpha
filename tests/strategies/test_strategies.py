import unittest

from async_timeout import asyncio
from alpha.strategies import get_all_strategies, run_backtest, find_file_by_strategy
from alpha.config import get_config_dir
import cfg4py
import datetime
import os
import omicron
from pyemit import emit
from alpha.core.const import E_BACKTEST


class StrategiesTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        os.environ[cfg4py.envar] = "DEV"
        cfg = cfg4py.init(get_config_dir())
        await omicron.init()
        await emit.start(emit.Engine.REDIS, start_server=True, dsn=cfg.redis.dsn)
        return await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        await emit.stop()
        await omicron.close()

        return await super().asyncTearDown()

    def test_get_all_strategies(self):
        cfg4py.init(get_config_dir())
        actual = get_all_strategies()

        for item in actual:
            if item[0] == "grid-strategy":
                break
        else:
            self.fail("GridStrategy not found.")

    def test_find_file_by_strategy(self):
        file = find_file_by_strategy("sma")
        print(file)

    async def test_run_backtest(self):
        strategy = "sma"
        start = datetime.date(2022, 1, 1)
        end = datetime.date(2022, 1, 20)
        params = {"code": "000001.XSHE", "frame_type": "1d"}

        state = {"stopped": False}

        async def foo(state, event):
            print("==============foo================\n", event)
            if event["event"] == "finished":
                state["stopped"] = True

        from functools import partial

        emit.register(E_BACKTEST, partial(foo, state))
        await run_backtest(strategy, start, end, params=params)
        while state["stopped"] is False:
            await asyncio.sleep(0.1)
