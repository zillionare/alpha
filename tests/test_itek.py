import unittest

from omicron.core.lang import async_run

from alpha.notify.itek import ItekClient


class MyTestCase(unittest.TestCase):
    @async_run
    async def test_something(self):
        itek = ItekClient()
        path = await itek.tts("这是tts单元测试", "alpha_ut.mp3")
        self.assertEqual(path, "/tmp/alpha_ut.mp3")



if __name__ == '__main__':
    unittest.main()


