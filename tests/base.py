import os
import unittest

import cfg4py
import omicron

from alpha.config import get_config_dir
from omicron.core.lang import async_run


class AbstractTestCase(unittest.TestCase):
    @async_run
    async def setUp(self) -> None:
        os.environ[cfg4py.envar] = 'DEV'
        config_dir = get_config_dir()
        cfg4py.init(config_dir)

        await omicron.init()