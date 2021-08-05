from pytest import param
from alpha.strategies.z04 import Z04
from tests import init_test_env
import unittest
import omicron
from scipy.stats import randint, uniform


class TestZ04(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        await omicron.init()
        return await super().asyncSetUp()

    async def test_build_dataset(self):
        s = Z04()
        await s.build_dataset("002176.XSHE", 2500, "/tmp/z04.ds.pkl")

    def test_grid_search(self):
        s = Z04()
        s.load_data("/tmp/z04.ds.pkl")

        params = {
            "colsample_bytree": uniform(0.7, 0.3),
            "gamma": uniform(0, 0.5),
            "learning_rate": uniform(0.03, 1),
            "max_depth": randint(2, 6),
            "n_estimators": randint(100, 150),
            "subsample": uniform(0.6, 0.4),
        }
        s.grid_search(params)
