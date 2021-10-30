from typing import List
from alpha.strategies.z05 import Z05
from alpha.core import Frame
from omicron.core.timeframe import tf
from omicron.models.securities import Securities
from omicron.models.security import Security
import arrow
from omicron.core.types import FrameType
import numpy as np
from alpha.core.features import fillna, relative_strength_index
from alpha.core.rsi_stats import rsiday, rsi30


class Z51(Z05):
    """在Z05的基础上，改变搜索和开仓条件，即先按30分钟搜索，如果RSI处于低位，则再看前一日是否符合日线买入条件。符合则开仓。

    Args:
        Z05 ([type]): [description]
    """



