import lightgbm as lgb
import numpy as np

from alpha.strategies.ml_strategy_base import MLStrategyBase


class TheForceStrategy(MLStrategyBase):
    """更加注重量能在股份涨跌中的作用的机器学习策略"""

    @classmethod
    async def build_dataset(cls, name: str, *args, **kwargs) -> np.ndarray:
        """构建数据集

        Args:
            name: 名称
            *args:
            **kwargs:

        Returns:
            数据集
        """

    def get_train_data(self, code: str):
        pass

    def m1_features(
        self, m1bars: np.ndarray, ref_bars: np.ndarray, prevday_m1bars: np.ndarray
    ) -> np.ndarray:
        """提取当天分钟线特征


        Args:
            m1bars : 载止当前的当天所有的分钟线
            ref_bars: 参考行情，一般为指数
            prevday_m1bars: 前一天的分钟线

        Returns:
            分钟线特征
        """
        pass

    def day_features(self, bars: np.ndarray) -> np.ndarray:
        """日线级别特征

        Args:
            bars : 日线数据

        Returns:
            一维numpy数组
        """
        pass
