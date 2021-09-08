from alpha.features.stockmath import find_runs
import unittest

class TestStockMatch(unittest.TestCase):
    def test_find_runs(self):
        data = [0, 0, 1, 2, 3, 3, 3,4]

        result = find_runs(data)
        self.assertListEqual([0, 1, 2, 3, 4], result[0].tolist())
        self.assertListEqual([0, 2, 3, 4, 7], result[1].tolist())
        self.assertListEqual([2, 1, 1, 3, 1], result[2].tolist())
