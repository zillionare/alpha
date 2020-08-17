import os
import unittest

import cfg4py
import omicron
from pyemit import emit

from alpha.config import get_config_dir
from omicron.core.lang import async_run

cfg = cfg4py.get_instance()
class AbstractTestCase(unittest.TestCase):
    @async_run
    async def setUp(self) -> None:
        os.environ[cfg4py.envar] = 'DEV'
        config_dir = get_config_dir()
        cfg4py.init(config_dir,False)

        await omicron.init()
        await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn)
