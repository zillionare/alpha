import unittest
from unicodedata import decimal

import numpy as np
from alpha.core.smvecstore import SmallSizeVectorStore


class TestSmallSizeVectorStore(unittest.TestCase):
    """Unit tests for the `SmVecStore` class."""

    def setUp(self):
        pass

    def test_all_in_one(self):
        store = SmallSizeVectorStore("test", {"desc": "O", "flag": "<i4"})

        rid = store.insert({"flag": 1, "desc": "one"}, [1, 2, 3])
        self.assertEqual(rid, 0)

        rids = store.insert(
            [{"flag": -1, "desc": "two"}, {"flag": -1, "desc": "three"}],
            [[1, 2, 0], [1, 0, 3]],
        )
        self.assertListEqual(rids, [1, 2])

        meta = store[2]
        self.assertEqual(meta["flag"], -1)

        rows = store[[0, 1]]
        self.assertEqual(len(rows), 2)
        self.assertListEqual([1, -1], rows["flag"].tolist())

        matched = store.search_vec([1, 2, 3], 999, n=3)
        self.assertListEqual([1, -1, -1], matched["flag"].tolist())
        np.testing.assert_array_almost_equal([0, 2, 3], matched["d"].tolist())

        matched = store.search_vec([1, 2, 3], 20, metric="Cosine", n=3)
        np.testing.assert_array_almost_equal(
            [0, 0.15, 0.4], matched["d"].tolist(), decimal=2
        )

        store.save("/tmp/smvec.pkl")
        store2 = SmallSizeVectorStore.load("/tmp/smvec.pkl")

        self.assertEqual(len(store2), len(store))
        self.assertListEqual([1, -1, -1], store2["flag"].tolist())

        vec, meta = store.get_vectors("flag", 1)
        self.assertListEqual([1.0, 2.0, 3.0], vec.flatten().tolist())

        store.remove("flag", 1)
        self.assertEqual(2, len(store))
        store.remove("flag", -1, {"desc": "two"})
        self.assertListEqual([1.0, 0.0, 3.0], store.vectors[0].tolist())
