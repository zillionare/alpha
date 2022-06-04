import asyncio
from typing import Any


class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance


class DataEvent(asyncio.Event):
    """An event object which carries data

    asyncio.Event is good for synchronization among coroutines, but it lacks the ability to share data between coroutines.

    """

    def __init__(self, data=None):
        super().__init__()
        self._data = data

    @property
    def data(self):
        return self._data

    def set(self, data: Any):
        self._data = data
        super().set()
