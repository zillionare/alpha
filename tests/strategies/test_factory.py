import logging
import unittest

from alpha.strategies.z01 import Z01Strategy
from alpha.strategies import factory

logger = logging.getLogger(__name__)


class TestStrategyFactory(unittest.TestCase):
    def test_register(self):
        factory.register("test", "test.TestStrategy")

        try:
            factory.register("test", "test.TestStrategy")
            self.assertTrue(False, "Should not go here")
        except ValueError:
            self.assertTrue(True)

        self.assertEquals(factory.reg.get("test"), "test.TestStrategy")

    def test_create_strategy(self):
        z01 = factory.create_strategy("z01", 8, f1=4, f2=5)
        self.assertTrue(isinstance(z01, Z01Strategy))
