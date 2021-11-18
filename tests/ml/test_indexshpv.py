import unittest

import omicron

from alpha.ml.index_sh_pv import IndexShPeakValleys
from tests import init_test_env


class TestIndexShPeakValleys(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        await omicron.init()
        return await super().asyncSetUp()

    async def test_index_sh_peak_valley(self):
        pv = IndexShPeakValleys()

        await pv.make_dataset()
        pv.train()

    async def test_watch(self):
        pv = IndexShPeakValleys(inference_mode=True)

        label, desc = await pv.watch()
        print(label, desc)
