"""small size vector store"""

from asyncio.log import logger
from os import stat
import pickle
from sqlite3 import DataError
from typing import Any, List, Union
from black import E
import numpy as np
import logging
import pandas as pd

from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SmallSizeVectorStore:
    def __init__(self, name: str, columns: dict, vector_type: str = "<f4"):
        self.name = name
        self.vector_type = vector_type

        # maintain the order of columns
        self.colnames = list(columns.keys())
        self.columns = columns

        self.vectors = None
        self.meta = None

    def _insert_one(self, meta_item: dict, vector):
        meta = np.array(
            [tuple(meta_item[col] for col in self.colnames)],
            dtype=[(col, self.columns[col]) for col in self.colnames],
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

    def remove(self, key: str, value: Any, **mapping):
        if key is None and not mapping:
            raise DataError("at least one key/value pair must exists")

        condition = self.meta[key] == value
        if mapping:
            for k, v in mapping.items():
                condition = condition & (self.meta[k] == v)

        idx = np.argwhere(condition).flatten()

        try:
            meta = np.delete(self.meta, idx)
            vectors = np.delete(self.vectors, idx, axis=0)
        except Exception as e:
            raise e

        self.meta = meta
        self.vectors = vectors

        return idx

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

    def search_vec(
        self, vec: Union[np.array, List], threshold, n: int = 1, metric="L2"
    ):
        """search vector in store

        supported metrics are `L2`, `Cosine`
        """

        if isinstance(vec, list):
            vec = np.array(vec)

        if metric == "L2":
            d = euclidean_distances(self.vectors, vec.reshape(1, -1)).flatten()

        elif metric == "Cosine":
            d = cosine_distances(self.vectors, vec.reshape(1, -1)).flatten()

        pos_all_lt_threshold = (np.argwhere(d < threshold)).flatten()
        d_lt_threshold = d[pos_all_lt_threshold]
        indices = pos_all_lt_threshold[np.argsort(d_lt_threshold)[:n]]

        meta = self[indices]
        res_type = np.dtype(meta.dtype.descr + [("d", "<f4")])
        res = np.empty(meta.shape, dtype=res_type)
        for col in meta.dtype.names:
            res[col] = meta[col]
        res["d"] = d.flatten()[indices]

        return res

    def get_vectors(self, col: str, value: Any):
        """locate and return vectore by `col`"""
        if col not in self.columns:
            raise ValueError(f"{col} is not a valid column")

        idx = self.meta[col] == value
        return self.vectors[idx], self.meta[idx]

    def show_samples(self):
        """turn `self.meta` as a dataframe"""
        return pd.DataFrame(self.meta.tolist(), columns=self.colnames)