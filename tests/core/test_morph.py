from omicron.core.types import FrameType
from alpha.core.morph import MorphFeatures
from tests import init_test_env
import unittest
import omicron


class TestMorphFeatures(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_test_env()
        await omicron.init()

    async def test_morph_features(self):
        morph = MorphFeatures(FrameType.DAY)
        code = "300688.XSHE"
        id_ = await morph.add_morph_pattern(
            code,
            "20210830",
        )

        self.assertEqual(0, id_[0])

        id_ = await morph.add_morph_pattern(code, "20210830")

        id_ = await morph.add_morph_pattern(code, "20210827")

        morph.save_store("/tmp/morph_test.pkl")
        morph.load_store(path="/tmp/morph_test.pkl")
        id_ = await morph.add_morph_pattern(code, "20210827")
        self.assertEqual(1, id_[0])
