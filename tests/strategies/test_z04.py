from pytest import param
from alpha.strategies.z04 import Z04
from tests import init_test_env
import unittest
import omicron


class TestZ04(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        await omicron.init()
        return await super().asyncSetUp()

    async def test_build_dataset(self):
        s = Z04()
        await s.make_dataset("002176.XSHE", 2500, "/tmp/z04.ds.pkl")

    def test_grid_search(self):
        s = Z04()
        s.load_data("/tmp/z04.ds.pkl")

        s.fit()
