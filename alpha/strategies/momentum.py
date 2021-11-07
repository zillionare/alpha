"""通过动能转向来确定股票买卖点。

    将动能定义为股价（或者移动平均值）的二阶导。显然，当动能大于0时，股价会向上拐头；当动能小于0时，股价会向下拐头。如果这种趋势能够延续一段时间，则我们就可以通过动能的方向来预测股价运动方向。
"""
import numpy as np
from omicron.core.numpy_extensions import find_runs


class Momentum(object):
    def __init__(self, threshold=1e-3):
        """[summary]

        Args:
            threshold ([type], optional): 当动能变化绝对值小于此值时，忽略信号. Defaults to 1e-3.
        """
        self.threshold = threshold

    def get_signal(self, price):
        """获取买卖信号

        Args:
            price (list): 股价数据

        Returns:
            list: 买卖信号
        """
        d1 = price[1:] / price[:-1] - 1
        d2 = np.diff(d1)

        isup, pos, length = find_runs(d2 >= 0)

        # 只保留被认为是变化显著的点
        valid_pos = pos[abs(d2[pos]) > self.threshold]
        isup = isup[abs(d2[pos]) > self.threshold]
        pos = valid_pos

        # 转向信号从坐标上看落后price 2个frame, 动能转向后，股价还会继续运行一小段时间，比如所一个周期，因此，我们这里将其调整到
        up = pos[isup] + 1
        down = pos[~isup] + 1
