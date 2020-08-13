import unittest

import arrow
from alpha.plots.crossyear import CrossYear
from omicron.core.lang import async_run

from tests.base import AbstractTestCase


class MyTestCase(AbstractTestCase):
    @async_run
    async def test_scan(self):
        end = arrow.get("2020-8-7")
        plot = CrossYear()
        await plot.scan(end)


if __name__ == '__main__':
    unittest.main()
