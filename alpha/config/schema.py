# noqa
from typing import Optional


class Config(object):
    __access_counter__ = 0

    def __cfg4py_reset_access_counter__(self):
        self.__access_counter__ = 0

    def __getattribute__(self, name):
        obj = object.__getattribute__(self, name)
        if name.startswith("__") and name.endswith("__"):
            return obj

        if callable(obj):
            return obj

        self.__access_counter__ += 1
        return obj

    def __init__(self):
        raise TypeError("Do NOT instantiate this class")

    class redis:
        dsn: Optional[str] = None

    class postgres:
        enabled: Optional[bool] = None

    class omega:
        class urls:
            quotes_server: Optional[str] = None

    class alpha:
        data_home: Optional[str] = None

    class milvus:
        host: Optional[str] = None

        port: Optional[int] = None

        meta: Optional[str] = None
