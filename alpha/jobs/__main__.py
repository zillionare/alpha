import asyncio
import os
from typing import Coroutine

import arrow
import cfg4py
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import omicron
from multiprocessing import Process
from alpha.strategies.gridstrategy import GridStrategy
from alpha.config import get_config_dir


async def backtest_entry(func: Coroutine):
    print("backtest_entry", os.getpid())
    cfg4py.init(get_config_dir())
    await omicron.init()

    await func()
    await omicron.close()


if __name__ == "__main__":
    from alpha.strategies.gridstrategy import GridStrategy

    cfg4py.init(get_config_dir())
    s = GridStrategy(is_backtest=True)
    start = arrow.get("2022-01-01").date()
    stop = arrow.get("2022-01-31").date()

    from alpha.remote.server import start

    p = Process(target=start)
    p.start()

    import rpyc
    import time

    time.sleep(3)
    conn = rpyc.connect("localhost", port=18861)
    print(conn.root.echo("hello"))

    conn.root.backtest("grid", start, stop, "000001.XSHE")
