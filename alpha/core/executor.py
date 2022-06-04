"""提供了简单但可靠的工作者进程机制。

无论是使用MP还是RyPC,都容易遇到无法序列化且难以查错的问题。本模块通过subprocess来创建全新子进程，父子进程之间通过pyemit来通信。
"""
import asyncio
import datetime
import functools
import logging
import os
import subprocess
import sys
from typing import Any

import cfg4py
import omicron
from pyemit import emit

from alpha.config import get_config_dir
from alpha.core.const import (
    E_BACKTEST,
    E_EXECUTOR_BACKTEST,
    E_EXECUTOR_ECHO_CHILD,
    E_EXECUTOR_ECHO_PARENT,
    E_EXECUTOR_EXIT,
    E_EXECUTOR_EXITED,
    E_EXECUTOR_STARTED,
)
from alpha.strategies import create_strategy_by_name

logger = logging.getLogger(__name__)
procs = set()


async def init():
    logger.info("init subprocess %s", os.getpid())
    cfg = cfg4py.init(get_config_dir())

    emit.register(E_EXECUTOR_BACKTEST, start_backtest)
    emit.register(E_EXECUTOR_EXIT, exit)
    emit.register(E_EXECUTOR_ECHO_CHILD, echo)
    await emit.start(emit.Engine.REDIS, start_server=True, dsn=cfg.redis.dsn)
    await omicron.init()
    await emit.emit(E_EXECUTOR_STARTED, os.getpid())


async def echo(msg):
    logger.info("executor %s recieved %s", os.getpid(), msg)
    await emit.emit(E_EXECUTOR_ECHO_PARENT, f"{msg} from {os.getpid()}")


async def exit(msg: Any = None):
    logger.info("exiting child process: %s", os.getpid())
    await omicron.close()
    await emit.emit(E_EXECUTOR_EXITED, os.getpid())
    await emit.stop()
    loop = asyncio.get_event_loop()
    loop.stop()


async def start_backtest(params: dict):
    """handler for event `E_EXECUTOR_BACKTEST`

    Args:
        params: contains the following keys:
            - strategy str: required, the strategy name, defined as BaseStrategy.name. required.
            - start datetime.date: required, the start date of backtest, required
            - end datetime.date: required, the end date of backtest, required
            - principal float: optional, default is 1_000_000,
            - params dict: optional, default is {}, additional params required by `strategy` during backtest
            - account str: optional, used to construct trader client
            - token str: optional, used to construct trader client
    """
    strategy = params.get("strategy")
    start = params.get("start")
    end = params.get("end")
    principal = params.get("principal", 1_000_000)
    params = params.get("params", {})
    account = params.get("account")
    token = params.get("token")

    assert all([strategy, start, end]), "strategy, start, end are required"

    try:
        s = create_strategy_by_name(strategy)
        if s is None:
            logger.warning("strategy %s not found", strategy)

        return await s.start_backtest(start, end, principal, params, account, token)
    except Exception as e:
        await emit.emit(
            E_BACKTEST,
            {
                "event": "failed",
                "strategy": strategy,
                "account": account,
                "token": token,
                "start": start,
                "end": end,
                "msg": str(e),
            },
        )


async def on_child_started(procs: set, pid: int):
    logger.info("child process %s started", pid)
    procs.remove(pid)


async def create_process_pool(n: int = 2):
    wait_start = set()

    emit.register(E_EXECUTOR_STARTED, functools.partial(on_child_started, wait_start))
    script = os.path.join(os.path.dirname(__file__), "executor.py")
    cmd = [sys.executable, script]

    for _ in range(n):
        proc = subprocess.Popen(cmd)
        wait_start.add(proc.pid)

    logger.info("waiting for executors starting: %s", wait_start)
    timeout = 30
    while len(wait_start) > 0 and timeout > 0:
        await asyncio.sleep(1)
        timeout -= 1


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(init())
    loop.run_forever()
    loop.close()
