import logging
import unittest

from alpha.core import strategyfactory
from alpha.strategies.z01 import Z01Strategy

logger = logging.getLogger(__name__)


class TestStrategyFactory(unittest.TestCase):
    def test_register(self):
        strategyfactory.register("test", "test.TestStrategy")

        try:
            strategyfactory.register("test", "test.TestStrategy")
            self.assertTrue(False, "Should not go here")
        except ValueError:
            self.assertTrue(True)

        self.assertEquals(strategyfactory.reg.get("test"), "test.TestStrategy")

    def test_create_strategy(self):
        z01 = strategyfactory.create_strategy("z01", 8, f1=4, f2=5)
        self.assertTrue(isinstance(z01, Z01Strategy))
