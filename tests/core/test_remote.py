import logging
import unittest
from tests import init_test_env
from alpha.core.remote import RemoteService
from alpha.strategies import run_backtest
import datetime
from pyemit import emit

logger = logging.getLogger(__name__)


class TestRemote(unittest.TestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        return super().asyncSetUp()

    async def test_remote_call(self):
        async def init():
            import cfg4py
            import omicron

            from alpha.config import get_config_dir

            logger.info("init is called")
            cfg4py.init(get_config_dir())

            await omicron.init()

        async def close():
            import omicron

            logger.info("close is called.")
            await omicron.close()

        rs = RemoteService(on_connect=init, on_disconnect=close)
        remote_call = rs.remote_call

        def no_args():
            return "no_args"

        actual = remote_call(no_args)
        self.assertEqual("no_args", actual)

        def one_arg(msg):
            return msg

        actual = remote_call(one_arg, "with_args")
        self.assertEqual("with_args", actual)

        def two_args(msg, msg2):
            return msg + msg2

        actual = remote_call(two_args, "with_args", "and_args")
        self.assertEqual("with_argsand_args", actual)

        def with_kwargs(msg, name="name"):
            return msg, {"name": name}

        actual = remote_call(with_kwargs, "with_kwargs", name="with_kwargs")
        self.assertEqual("with_kwargs", actual[0])
        exp = {"name": "with_kwargs"}
        self.assertSetEqual(set(exp.keys()), set(actual[1].keys()))
        self.assertSetEqual(set(exp.values()), set(actual[1].values()))

        async def async_foo(*args, **kwarg):
            return args, kwarg

        future = remote_call(async_foo, "args", kwargs="kwargs")
        args, kwargs = future.result()
        self.assertEqual(("args",), args)
        self.assertSetEqual(set(["kwargs"]), set(kwargs.values()))

        # disable this test in unit test, as it will take too long
        async def long_call(msg):
            import asyncio

            await asyncio.sleep(10)
            return msg

        logger.info("execute and wait a long async call, it will cost 10 seconds")
        future = remote_call(long_call, "long_call")
        import time
        t0 = time.time()
        actual = future.result()
        cost = time.time() - t0
        self.assertTrue(abs(cost - 10) < 1)
        self.assertEqual("long_call", actual)

        async def test_cache():
            # this is for checking if omicron is init as required
            # if omicron is initiated, then it will return "_sys_"
            from omicron import cache

            return await cache.sys.get("__meta__.database")

        future = remote_call(test_cache)
        actual = future.result()
        self.assertEqual("_sys_", actual)

    async def test_remote_strategy(self):
        async def init():
            import cfg4py
            import omicron

            from alpha.config import get_config_dir

            logger.info("init is called")
            cfg4py.init(get_config_dir())

            await omicron.init()

        async def close():
            import omicron

            logger.info("close is called.")
            await omicron.close()

        async def on_notify(msg):
            print(msg)

        emit.register
        rs = RemoteService(on_connect=init, on_disconnect=close)
        remote_call = rs.remote_call

        start = datetime.date(2022, 3, 1)
        end = datetime.date(2022, 3, 30)
        params = {
            "frame_type": "1d",
            "code": "000001.XSHE"
        }
        future = remote_call(run_backtest, "sma", start, end, params = params)
        actual = future.result()
        pass
