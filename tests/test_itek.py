import unittest

from tests.base import AbstractTestCase

from alpha.core.enums import Events
from omicron.core.lang import async_run
from pyemit import emit
import cfg4py

from alpha.notify.itek import ItekClient

cfg = cfg4py.get_instance()
class MyTestCase(AbstractTestCase):
    @async_run
    async def test_something(self):
        itek = ItekClient('/notebooks/msg/')
        path = await itek.tts("这是tts单元测试", "alpha_ut.mp3")
        await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn)
        await emit.emit(Events.sig_long, {"desc":"这是单元测试"})
        #self.assertEqual(path, "/tmp/alpha_ut.mp3")



if __name__ == '__main__':
    unittest.main()


