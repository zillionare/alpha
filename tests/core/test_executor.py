import asyncio
import unittest

from pyemit import emit

from alpha.core.const import (
    E_EXECUTOR_ECHO_PARENT,
    E_EXECUTOR_ECHO_CHILD,
    E_EXECUTOR_EXIT,
    E_EXECUTOR_EXITED,
)
from alpha.core.executor import create_process_pool
from tests import init_test_env
from functools import partial


class ExecutorTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        cfg = init_test_env()

        async def on_echo_back(msg):
            print(msg)

        async def on_child_exit(event: asyncio.Event, pid):
            event.set()

        self.event = asyncio.Event()
        emit.register(E_EXECUTOR_EXITED, partial(on_child_exit, self.event))
        emit.register(E_EXECUTOR_ECHO_PARENT, on_echo_back)
        await emit.start(emit.Engine.REDIS, start_server=True, dsn=cfg.redis.dsn)

        return await super().asyncSetUp()

    async def test_start_backtest(self):
        await create_process_pool(1)
        await emit.emit(E_EXECUTOR_ECHO_CHILD, "hello")
        await emit.emit(E_EXECUTOR_EXIT)
        await self.event.wait()
        print("parent process exit")
