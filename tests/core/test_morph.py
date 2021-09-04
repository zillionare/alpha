from re import T
import unittest

import omicron
from omicron.core.types import FrameType

from alpha.core.morph import MorphFeatures
from tests import init_test_env


class TestMorphFeatures(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        await omicron.init()

    async def test_morph_features(self):
        morph = MorphFeatures(FrameType.DAY, flen=20, thresholds={
            5: 1e-3,
            10: 1e-3,
            20: 1e-3,
            60: 5e-4,
        })
        code = "300688.XSHE"
        features = await morph.encode(
            code,
            "20210830",
        )

        self.assertListEqual([0,0,0,0], features)

        features = await morph.encode(code, "20210830")
        self.assertListEqual([0,0,0,0], features)

        features = await morph.encode(code, "20210827")
        self.assertListEqual([1,1,1,0], features)

        morph.dump("/tmp/morph_test.pkl")
        morph.load(path="/tmp/morph_test.pkl")
        features = await morph.encode(code, "20210809")
        self.assertEqual([2,2,2,1], features)
