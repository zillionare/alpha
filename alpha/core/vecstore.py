from alpha.config import get_config_dir
from typing import List, Union
from pymilvus import DataType, Milvus
from pymongo import MongoClient
import cfg4py
import numpy as np
import logging

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()

class VecStore:
    def __init__(
        self,
        name: str,
        metric: str,
        dim: int,
        milhost: str,
        milport: int,
        meta_dsn: str = None,
    ) -> None:
        """large-scale vectores store, which store vectors in milvus and meta data in mongo database

        Args:
            name (str): the name used by both milvus collection and metadata table (SQL)
            milhost (str): [description]
            milport (int): [description]
            meta_dsn (str, optional): dsn for mongo, if there's meta data
        """
        self.name = name
        self.milvus = Milvus(host=milhost, port=milport)

        if self.milvus.has_collection(name):
            self.milvus.load_collection(self.name, timeout=10)

        if meta_dsn:
            self.mongo = MongoClient(meta_dsn)
            self.meta_collection = self.mongo["alpha"][self.name]
        else:
            self.meta_collection = None

        self.metric = metric
        self.dim = dim

    def create_collection(self, drop_if_exists=True):
        if drop_if_exists:
            if self.milvus.has_collection(self.name):
                self.milvus.drop_collection(self.name)
            if self.meta_collection:
                self.meta_collection.remove({})
        else:
            if self.milvus.has_collection(self.name):
                return

        id_field = {
            "name": "_id",
            "type": DataType.INT64,
            "auto_id": True,
            "is_primary": True,
        }

        features_field = {
            "name": "features",
            "type": DataType.FLOAT_VECTOR,
            "metric_type": self.metric,
            "params": {"dim": self.dim},
            "indexes": [{"metric_type": self.metric}],
        }

        self.milvus.create_collection(self.name, {"fields": [id_field, features_field]})
        self.milvus.load_collection(self.name)

    def insert(self, vecs: Union[List, List[List]], meta: List[dict] = None):
        """[summary]

        Args:
            vecs (Union[List, List[List]]): [description]
            meta (dict, optional):

        Returns:
            the id of the record, same as in both milvus and mongodb
        """
        assert isinstance(vecs, list)

        mr = self.milvus.insert(
            self.name,
            [{"name": "features", "type": DataType.FLOAT_VECTOR, "values": vecs}],
        )

        ids = mr.primary_keys

        if meta is not None:
            if len(meta) != len(ids):
                raise ValueError("the size of meta and vecs must be same")

        try:
            self.meta_collection.insert_many(
                ({"_id": _id, **meta_item}) for _id, meta_item in zip(ids, meta)
            )
        except Exception as e:
            logger.exception(e)
            # todo: delete recs in milvus

        return ids

    def search_vec(self, vecs: List[List], threshold: float, limit=2):
        """search the store by vector"""
        params = {"metric_type": self.metric, "params": {"nprobe": 10}}
        res = self.milvus.search_with_expression(
            self.name,
            vecs,
            "features",
            param=params,
            limit=limit,
            output_fields=["_id"]
        )

        ids, distances = [], []
        for i in range(len(res)): # chuncks
            ids.extend(res[i].ids)
            distances.extend(res[i].distances)

        dist_all = np.array(distances)
        pos = np.argwhere(dist_all < threshold)
        if len(pos) == 0:
            return None

        ids = np.array(ids)[pos].flatten()
        distances = {_id: d for _id, d in zip(ids, dist_all[pos].flatten())}

        if self.meta_collection is not None:
            meta_data = self.meta_collection.find({
                "_id": {
                    "$in" : ids.tolist()
                }
            })

        results = []
        for meta in meta_data:
            _id = meta["_id"]
            results.append({
                "distance": distances.get(_id),
                **meta
            })

            del distances[_id]

        if len(distances) > 0:
            # in case errors happened during inserting
            logger.warning("records %s has no meta bound", distances.keys())

        return results

    def search_by_meta(self, return_vec=False, **query):
        metas = list(self.meta_collection.find(query))
        if len(metas) == 0:
            return None

        if not return_vec:
            return metas

        results = []
        ids = [meta["_id"] for meta in metas]

        res = self.milvus.query(self.name, f"_id in {ids}", output_fields=["_id", "features"])
        features = {item["_id"]: item["features"] for item in res}

        for meta in metas:
            _id = meta["_id"]

            results.append({
                "features": features.get(_id),
                **meta
            })

        return results



