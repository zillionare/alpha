"""small size vector store"""

from asyncio.log import logger
from os import stat
import pickle
from typing import Any, List, Union
import numpy as np
import logging

from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SmallSizeVectorStore:
    def __init__(self, name: str, columns: dict, vector_type: str = "<f4"):
        self.name = name
        self.vector_type = vector_type

        # maintain the order of columns
        self.cols = list(columns.keys())
        self.columns = columns

        self.vectors = None
        self.meta = None

    def _insert_one(self, meta_item: dict, vector):
        meta = np.array(
            [tuple(meta_item[col] for col in self.cols)],
            dtype=[(col, self.columns[col]) for col in self.cols],
        )

        try:
            if self.vectors is None:
                vectors_ = np.array([vector], dtype=self.vector_type)
                meta_ = meta
            else:
                vectors_ = np.vstack((self.vectors, vector))
                meta_ = np.concatenate((self.meta, meta))
        except Exception as ex:
            logger.error(f"failed to insert vector {vector} to store {self.name}")
            raise ex

        self.vectors = vectors_
        self.meta = meta_

        return len(self.vectors) - 1

    def insert(self, meta_items: Union[dict, List[dict]], vectors):
        """insert item to store"""
        item = meta_items if isinstance(meta_items, dict) else meta_items[0]
        diff = set(self.columns.keys()) - set(item.keys())
        if diff:
            raise ValueError(f"these columns {diff} are required")

        if isinstance(meta_items, dict):
            return self._insert_one(meta_items, vectors)

        ids = []
        if len(meta_items) != len(vectors):
            raise ValueError(f"length of meta and vectors should be same")

        for item, vector in zip(meta_items, vectors):
            ids.append(self._insert_one(item, vector))

        return ids

    def __getitem__(self, indice: Union[int, List]):
        return self.meta[indice]

    def __len__(self):
        return len(self.meta)

    def __iter__(self):
        for idx in range(len(self.meta)):
            yield self[idx]

    @staticmethod
    def load(path: str):
        """load store from file"""
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: str):
        """save store to file"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def search_vec(self, vec: Union[np.array, List], top_n=5, metric="L2"):
        """search vector in store

        supported metrics are `L2`, `Cosine`
        """

        if isinstance(vec, list):
            vec = np.array(vec)

        if metric == "L2":
            d = euclidean_distances(self.vectors, vec.reshape(1, -1))
            indices = np.argsort(d.flatten())[:top_n]

        if metric == "Cosine":
            d = cosine_distances(self.vectors, vec.reshape(1, -1))
            indices = np.argsort(d.flatten())[:top_n]

        meta = self[indices]
        res_type = np.dtype(meta.dtype.descr + [("d", "<f4")])
        res = np.empty(meta.shape, dtype=res_type)
        for col in meta.dtype.names:
            res[col] = meta[col]
        res["d"] = d.flatten()

        return res

    def get_vectors(self, col:str, value: Any):
        """locate and return vectore by `col`"""
        if col not in self.columns:
            raise ValueError(f"{col} is not a valid column")

        idx = self.meta[col] == value
        return self.vectors[idx], self.meta[idx]
