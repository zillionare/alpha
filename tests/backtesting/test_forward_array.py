import unittest

import numpy as np
from alpha.backtesting.forward_array import ForwardArray


class TestForwardArray(unittest.TestCase):
    def test_str(self):
        data = np.array([i for i in range(3)], dtype=[("seq", "<i4")])
        fa = ForwardArray(data)

        expected = "<ForwardArray>:unnamed@3\n[(0,) (1,) (2,)]"
        self.assertEqual(expected, str(fa))

    def test_repr(self):
        data = np.array([i for i in range(3)], dtype=[("seq", "<i4")])
        fa = ForwardArray(data)

        expected = "<ForwardArray>:unnamed@3"
        self.assertEqual(expected, repr(fa))

    def test_len(self):
        data = np.array([i for i in range(3)], dtype=[("seq", "<i4")])
        fa = ForwardArray(data, start_pos=2)

        self.assertEqual(2, len(fa))

    def test_index_accessor(self):
        data = np.array([i for i in range(3)], dtype=[("seq", "<i4")])
        fa = ForwardArray(data, start_pos=2)

        self.assertListEqual([(0,), (1,)], fa[:2].tolist())
        self.assertListEqual([(0,), (1,)], fa[:3].tolist())
        self.assertTupleEqual((0,), tuple(fa[0]))

        fa[0][0] = 1
        self.assertEqual([(1,)], fa[:1].tolist())

        fa = ForwardArray(data, start_pos=3)

        fa[:2] = fa[1:3]
        self.assertListEqual([(1,), (2,), (2,)], fa[:3].tolist())

    def test_reveal(self):
        data = np.array([i for i in range(3)], dtype=[("seq", "<i4")])
        fa = ForwardArray(data)

        self.assertEqual(0, len(fa))
        fa.reveal()

        self.assertEqual(1, len(fa))

        fa.reveal(2)
        self.assertEqual(3, len(fa))

    def test_set_pos(self):
        data = np.array([i for i in range(3)], dtype=[("seq", "<i4")])
        fa = ForwardArray(data, start_pos=2)

        fa.set_pos(1)
        self.assertEqual(1, len(fa))

    def test_add_feature(self):
        data = np.array([i for i in range(3)], dtype=[("seq", "<i4")])
        fa = ForwardArray(data, start_pos=2)

        feature = [np.nan, np.nan, 1]
        fa.add_feature("mean", feature, dtype="<f4", valid_pos=2)

        expected = [(0, np.nan), (1, np.nan), (2, 1.0)]
        np.testing.assert_allclose(expected, fa.data.tolist())
