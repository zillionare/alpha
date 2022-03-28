import unittest
from tests import init_test_env
import omicron
from alpha.features.tradelimits import trade_limits


class TradeLimitsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        await omicron.init()
        return await super().asyncSetUp()

    async def test_buy_limit_price(self):
        code = "600156.XSHG"

        r = await trade_limits(code, 10)
        print(r)
