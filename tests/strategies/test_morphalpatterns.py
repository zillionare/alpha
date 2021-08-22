
import time
from alpha.core.morphalpatterns import MorphaFeatures
import unittest
import numpy as np


class TestMorphalPatterns(unittest.TestCase):
    def test_morphalpatterns(self):
        mf = MorphaFeatures()

        stores = {5: mf.store_ma5, 10: mf.store_ma10, 20: mf.store_ma20}
        for win in [5, 10, 20]:
            print(stores[win].show_samples())

    def test_xtransform(self):
        t0 = time.time()
        mf = MorphaFeatures()
        print(time.time() - t0)

        p = np.poly1d((0.0002, 0.005, 1))
        y = p(np.arange(mf.nbars))
        features = mf.xtransform(y)

        self.assertEqual(15, len(features))
        np.testing.assert_array_almost_equal([0.00095, 0.012, 0.2737], features[:3], decimal=3)


