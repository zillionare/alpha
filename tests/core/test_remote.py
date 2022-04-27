import logging
import unittest
from tests import init_test_env
from alpha.core.remote import RemoteService, remote_call

logger = logging.getLogger(__name__)


class TestRemote(unittest.TestCase):
    def setUp(self) -> None:
        init_test_env()
        return super().setUp()

    def test_remote_call(self):
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

        RemoteService.add_event_listener("on_connect", init)
        RemoteService.add_event_listener("on_disconnect", close)

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

        args, kwargs = remote_call(async_foo, "args", kwargs="kwargs")
        self.assertEqual(("args",), args)
        self.assertSetEqual(set(["kwargs"]), set(kwargs.values()))

        # disable this test in unit test, as it will take too long
        # async def long_call(msg):
        #     import asyncio

        #     await asyncio.sleep(240)
        #     return msg

        # actual = remote_call(long_call, "long_call")
        # self.assertEqual("long_call", actual)

        async def test_cache():
            # this is for checking if omicron is init as required
            # if omicron is initiated, then it will return "_sys_"
            from omicron import cache

            return await cache.sys.get("__meta__.database")

        actual = remote_call(test_cache)
        self.assertEqual("_sys_", actual)
