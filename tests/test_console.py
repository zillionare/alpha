import unittest

from omicron.core.lang import async_run

from tests.base import AbstractTestCase


class MyTestCase(AbstractTestCase):
    @async_run
    async def test_list_monitors(self):
        from alpha.core.console import list_monitors

        await list_monitors(plot='maline')

    @async_run
    async def test_list_stock_pool(self):

        from alpha.core.console import list_momentum_pool
        await list_momentum_pool()

    @async_run
    async def test_list_jobs(self):
        from alpha.core.console import list_jobs

        await list_jobs()



if __name__ == '__main__':
    unittest.main()
