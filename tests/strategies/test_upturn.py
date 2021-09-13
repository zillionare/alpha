from alpha.strategies.upturn import UpTurn
import unittest
import cfg4py
import omicron
from alpha.config import get_config_dir

class TestUpTurn(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        cfg4py.init(get_config_dir())
        await omicron.init()
        return await super().asyncSetUp()

    async def asyncTearDown(self) -> None:
        pass

    async def test_upturn(self):
        """
        Test upturn
        """
        up = UpTurn()
        await up.scan('2021-09-13', ['000001.XSHE'], profit=0.07)

        print(up.to_dataframe(up.X))
