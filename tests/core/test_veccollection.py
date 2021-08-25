import unittest

import cfg4py
import numpy as np
from pymilvus import Milvus
from alpha.config import get_config_dir
from alpha.core.veccollection import VecCollection
from pymongo import MongoClient

class TestVecCollection(unittest.TestCase):
    def test_all_in_one(self):
        """to run this test, you need to setup a milvus server and a mongo server"""
        cfg = cfg4py.init(get_config_dir())
        milvus = Milvus(host=cfg.milvus.host, port=cfg.milvus.port)
        mongo = MongoClient(cfg.mongo.dsn)

        vc = VecCollection(
            "alpha_test", "L2", 50, milvus, mongo.alpha_test
        )
        vc.create_collection(drop_if_exists=True)

        vec = np.random.rand(10, 50)
        meta = [{"flag": i // 2, "seq": i} for i in range(10)]

        ids = vc.insert(vec.tolist(), meta)
        self.assertEqual(len(ids), 10)

        res = vc.search_vec(vec[:2].tolist(), 0.2)
        self.assertEqual(2, len(res))
        self.assertAlmostEqual(res[0]["distance"], 0.0)
        self.assertEqual(res[0]["flag"], 0)

        res = vc.search_by_meta(flag=1, seq=2, return_vec=True)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["flag"], 1)
        self.assertEqual(res[0]["seq"], 2)
        self.assertEqual(len(res[0]["features"]), 50)

        vc.drop_collection()
