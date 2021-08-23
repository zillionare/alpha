
import time
from alpha.core.morphalpatterns import MorphaFeatures
import unittest
import numpy as np


class TestMorphalPatterns(unittest.TestCase):
    def test_morphalpatterns(self):
        mf = MorphaFeatures()

        for win in [5, 10, 20]:
            print(mf.stores[win].show_samples())

    def test_xtransform(self):
        t0 = time.time()
        mf = MorphaFeatures()
        print(time.time() - t0)

        p = np.poly1d((0.0002, 0.005, 1))
        y = p(np.arange(mf.nbars))
        features = mf.xtransform(y)

        self.assertEqual(3, len(features))
        np.testing.assert_array_almost_equal([0.0002, 0.005, 0.0125], features[0].tolist()[:3], decimal=3)


