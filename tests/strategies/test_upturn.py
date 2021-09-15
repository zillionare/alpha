from alpha.strategies.upturn import UpTurn
import unittest
import cfg4py
import omicron
from alpha.config import get_config_dir
from omicron.models.securities import Securities


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
        await up.scan("2021-09-14 14:50", ["000001.XSHE"], profit=0.07)

        print(up.to_dataframe(up.X))

    async def test_plot(self):
        """
        Test upturn
        """
        up = UpTurn()
        codes = (Securities().choose(["stock"]))[:100]
        await up.scan("2021-09-13", codes, profit=0.0)
        df = up.to_dataframe(up.X)
        await up.plot(df)
