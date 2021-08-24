from alpha.core.vecstore import VecStore
from alpha.config import get_config_dir
import unittest
import numpy as np

import cfg4py
class TestVecStore(unittest.TestCase):
    def test_all_in_one(self):
        """to run this test, you need to setup a milvus server and a mongo server
        """
        cfg = cfg4py.init(get_config_dir())
        vs = VecStore("stock", "L2", 50, cfg.milvus.host, cfg.milvus.port, cfg.mongo.dsn)
        vs.create_collection(drop_if_exists=True)


        vec = np.random.rand(10, 50)
        meta = [{"flag": i//2, "seq": i} for i in range(10)]

        ids = vs.insert(vec.tolist(), meta)
        self.assertEqual(len(ids), 10)

        res = vs.search_vec(vec[:2].tolist(), 0.2)
        self.assertEqual(2, len(res))
        self.assertAlmostEqual(res[0]["distance"], 0.0)
        self.assertEqual(res[0]["flag"], 0)

        res = vs.search_by_meta(flag=1, seq=2, return_vec=True)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["flag"], 1)
        self.assertEqual(res[0]["seq"], 2)
        self.assertEqual(len(res[0]["features"]), 50)
