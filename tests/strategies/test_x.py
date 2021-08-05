from alpha.core.features import pos_encode
import itertools
import unittest
import numpy as np


class TextX(unittest.TestCase):
    def test_position_encoding(self):
        stationary = np.array([2, 4, 8, 16, 32, 64])
        results = []
        for composition in itertools.permutations(stationary):
            # (5, 10, 30, 20)
            results.append(
                {"val": pos_encode(stationary, composition), "comp": composition}
            )

        sorted_results = sorted(results, key=lambda x: x["val"])
        for r in sorted_results:
            print(f'{round(r["val"], 4)}:{r["comp"]}')
