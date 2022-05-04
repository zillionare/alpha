import unittest
from alpha.strategies import get_all_strategies
from alpha.config import get_config_dir
import cfg4py


class StrategiesTest(unittest.TestCase):
    def test_get_all_strategies(self):
        cfg4py.init(get_config_dir())
        actual = get_all_strategies()

        for item in actual:
            if item[2].__name__ == "GridStrategy":
                break
        else:
            self.fail("GridStrategy not found.")
