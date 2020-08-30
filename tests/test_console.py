import unittest

from omicron.core.lang import async_run

from tests.base import AbstractTestCase


class MyTestCase(AbstractTestCase):
    @async_run
    async def test_list_monitor(self):
        from alpha.core.console import list_monitors

        await list_monitors()



if __name__ == '__main__':
    unittest.main()
