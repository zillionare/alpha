import numpy as np
from typing import List
from typing import Tuple
import ckwrap


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


def plateaus(numbers: np.ndarray, min_size: int, fall_in_range_ratio: float = 0.97):
    """求数组`numbers`中的平台。

    如果一个数组中的多数元素（超过`fall_in_range_ratio`）都落在3个标准差以内，则认为该处有一个平台。

    Args:
        numbers: 输入数组
        min_size: 平台的最小长度
        fall_in_range_ratio: 平台的最小长度
    """
    clusters = clustering(numbers, len(numbers) // min_size)

    plats = []
    for (start, width) in clusters:
        if width < min_size:
            continue

        y = numbers[start : start + width]
        mean = np.mean(y)

        delta = np.std(y)

        inrange = len(y[(y <= (mean + 3 * delta)) & (y >= mean - 3 * delta)])
        ratio = inrange / len(y)

        if ratio >= fall_in_range_ratio:
            plats.append((start, width))

    return plats
