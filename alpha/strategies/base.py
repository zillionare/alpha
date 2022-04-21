from abc import ABCMeta, abstractmethod


class BaseStrategy(metaclass=ABCMeta):
    name = "base-strategy"
    desc = "Base Strategy Class"

    @abstractmethod
    def backtest(self, *args, **kwargs):
        pass
