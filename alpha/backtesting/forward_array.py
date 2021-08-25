from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from omicron.core.numpy_extensions import numpy_append_fields


class ForwardArray:
    def __init__(self, features: ArrayLike, start_pos: int = 1, name: str = None):
        """A forward array that reveals data step by step, work with `alpha.backtesting.Strategy`

        `features` is revealing along with `next` is called. You can add feature by calling `add_feature`, and these feature will be revealed incrementally too.

        `features` must be an numpy structured arrays, to cope with `alpha.backtesting.Stragety`, it must contain as least two fields: `frame` and `close`

        Args:
            features (ArrayLike): [description]
            start_pos (int): [description]
            name (str): the name of the features
        """
        if (not hasattr(features, "dtype")) or features.dtype.names is None:
            raise TypeError(
                "parameter `features` should be numpy structured arrays type"
            )

        self._features = features

        assert start_pos <= len(features)
        self._start_pos = start_pos
        self._end_pos = len(features)
        self._name = name

        self._cur = start_pos

    def __str__(self) -> str:
        n_row = min(3, len(self._features))
        rows = self._features[:n_row]

        name = self.__class__.__name__
        feature_name = self._name or "unnamed"

        return f"<{name}>:{feature_name}@{len(self._features)}\n{rows}"

    def __repr__(self) -> str:
        feature_name = self._name or "unnamed"
        return f"<{self.__class__.__name__}>:{feature_name}@{len(self._features)}"

    def __len__(self) -> int:
        return self._cur

    def __getattr__(self, name: str) -> Any:
        if name in self._features.dtype.names:
            return self._features[name][: self._cur]

    def __getitem__(self, subscript):
        return self._features[: self._cur][subscript]

    def __setitem__(self, subscript, item):
        self._features[: self._cur][subscript] = item

    def __delitem__(self, subscript):
        del self._features[: self._cur][subscript]

    @property
    def size(self):
        """how many items it actually contains

        Returns:
            [type]: [description]
        """
        return len(self._features)

    def reveal(self, n=1):
        self._cur += n
        return self._cur < self._end_pos

    def reset(self):
        self._cur = 0

    def set_pos(self, i: int):
        assert 0 <= i < len(self._features)
        self._cur = i

    def add_feature(self, name: str, data: np.array, dtype: str, valid_pos: int = 0):
        """add new feature

        Args:
            name (str): the field name of the new feature
            dtype (str): the dtype of the new feature, i.e, "<f4"
            data (np.array): [description]
            valid_pos (int, optional): [description]. Defaults to 0.
        """
        assert len(data) == len(self._features)

        _dtype = [(name, dtype)]
        self._features = numpy_append_fields(self._features, name, data, _dtype)

        self._cur = max(valid_pos, self._cur)

    @property
    def data(self):
        return self._features
