import datetime
import os
import pickle
import unittest

import numpy as np
from dateutil.tz import tzfile

from alpha.features.volume import top_volume_direction
from tests import data_dir


class TestVolumeFeatures(unittest.TestCase):
    def test_volume_features(self):
        # 600163.XSHG 2021-09-02 11:30

        bars = np.array(
            [
                (5.72, 5.74, 5.5, 5.61, 6753800.0, 37871375.0, 2.352),
                (5.62, 5.6900005, 5.5899997, 5.62, 2434500.0, 13724468.0, 2.352),
                (5.63, 5.76, 5.63, 5.71, 5078400.0, 28906977.0, 2.352),
                (5.7, 5.79, 5.63, 5.79, 4571400.0, 26181890.0, 2.352),
                (5.79, 5.86, 5.75, 5.82, 4766500.0, 27690079.0, 2.352),
                (5.82, 5.83, 5.74, 5.76, 4448900.0, 25807039.0, 2.352),
                (5.76, 5.94, 5.63, 5.94, 12006400.0, 69975416.0, 2.352),
                (5.95, 6.0600004, 5.89, 6.0, 13042900.0, 78035291.0, 2.352),
                (5.99, 6.0600004, 5.98, 6.05, 5927400.0, 35724998.0, 2.352),
                (6.04, 6.15, 5.99, 6.15, 6730000.0, 40837283.0, 2.352),
            ],
            dtype=[
                ("open", "<f4"),
                ("high", "<f4"),
                ("low", "<f4"),
                ("close", "<f4"),
                ("volume", "<f8"),
                ("amount", "<f8"),
                ("factor", "<f4"),
            ],
        )

        vr = top_volume_direction(bars, n=10)
        exp = [5.4, 0]
        np.testing.assert_array_almost_equal(exp, vr, 1)

        bars["volume"][5] *= 3
        vr = top_volume_direction(bars, n=10)
        exp = [5.4, -0.97]
        np.testing.assert_array_almost_equal(exp, vr, 1)
