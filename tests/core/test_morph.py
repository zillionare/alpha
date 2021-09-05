import unittest
from re import T
import numpy as np

from omicron.core.types import FrameType

from alpha.core.morph import MorphFeatures


class TestMorphFeatures(unittest.IsolatedAsyncioTestCase):
    def test_morph_features(self):
        morph = MorphFeatures(
            FrameType.DAY,
            flen=20,
            thresholds={
                5: 1e-3,
                10: 1e-3,
                20: 1e-3,
                60: 5e-4,
            },
        )

        # fmt: off
        close = [
            0.03, 0.73, 0.52, 0.71, 0.8 , 0.92, 0.09, 0.13, 0.03, 0.25, 0.24,
            0.53, 0.27, 0.19, 0.56, 0.34, 0.  , 0.45, 0.02, 0.46, 0.54, 0.26,
            0.04, 0.95, 0.89, 0.41, 0.4 , 0.25, 0.43, 0.97, 0.99, 0.35, 0.13,
            0.49, 0.12, 0.07, 0.91, 0.2 , 0.7 , 0.15, 0.7 , 0.96, 0.06, 0.2 ,
            0.44, 0.17, 0.11, 0.68, 0.39, 0.42, 0.9 , 0.12, 0.31, 0.86, 0.1 ,
            0.7 , 0.93, 0.96, 0.61, 0.23, 0.75, 0.89, 0.06, 0.72, 0.45, 0.56,
            0.29, 0.88, 0.49, 0.74, 0.31, 0.59, 0.81, 0.51, 0.33, 0.45, 0.  ,
            0.25, 0.92, 0.91, 0.88, 0.5 , 0.31, 0.6 , 0.57, 0.24, 0.49, 0.6 ,
            0.82, 0.95, 0.77, 0.09, 0.08, 0.71, 0.59, 0.05, 0.22, 0.63, 0.44,
            0.21, 0.18, 0.57, 0.49, 0.31, 0.84, 0.49, 0.65, 0.  , 0.53, 0.19,
            0.9 , 0.59, 0.71, 0.25, 0.  , 0.25, 0.78, 0.25, 0.49, 0.66
        ]

        close += [-1 * np.sin(i*0.2) for i in range(len(close))]
        # fmt: on

        features = morph.encode(close[:-20])

        self.assertListEqual([0, 0, 0, 0], features)

        features = morph.encode(close[:-20])
        self.assertListEqual([0, 0, 0, 0], features)

        features = morph.encode(close)
        self.assertListEqual([1, 1, 1, 1], features)

        morph.dump("/tmp/morph_test.pkl")
        morph.load(path="/tmp/morph_test.pkl")
        features = morph.encode(close[:-5])
        self.assertEqual([2,2,2,2], features)

        morph.dump("/tmp/morph_test.pkl")
        morph.load(path="/tmp/morph_test.pkl")
        self.assertEqual(3, morph.version)

        # sorted so the id of feature changes
        features = morph.encode(close[:-5])
        self.assertEqual([1, 0, 0, 1], features)
