from typing import List, Tuple

import ckwrap
import numpy as np


class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance


def clustering(numbers: np.ndarray, n: int) -> List[Tuple[int, int]]:
    """将数组`numbers`划分为`n`个簇

    Examples:
        >>> numbers = np.array([1,1,1,2,4,6,8,7,4,5,6])
        >>> clustering(numbers, 2)
        [(0, 4), (4, 7)]
    """
    result = ckwrap.cksegs(numbers, n)

    clusters = []
    for pos, size in zip(result.centers, result.sizes):
        clusters.append((int(pos - size // 2 - 1), int(size)))

    return clusters


def plateaus(
    numbers: np.ndarray, min_size: int, fall_in_range_ratio: float = 0.97
) -> List[Tuple]:
    """求数组`numbers`中的平台。

    如果一个数组中的多数元素（超过`fall_in_range_ratio`）都落在3个标准差以内，则认为该处有一个平台。

    Args:
        numbers: 输入数组
        min_size: 平台的最小长度
        fall_in_range_ratio: 平台的最小长度

    Returns:
        平台的起始位置和长度
    """
    if numbers.size <= min_size:
        n = 1
    else:
        n = numbers.size // min_size

    clusters = clustering(numbers, n)

    plats = []
    for (start, length) in clusters:
        if length < min_size:
            continue

        y = numbers[start : start + length]
        mean = np.mean(y)

        delta = np.std(y)

        inrange = len(y[abs(y - mean) <= 3 * delta])
        ratio = inrange / length

        if ratio >= fall_in_range_ratio:
            plats.append((start, length))

    return plats
