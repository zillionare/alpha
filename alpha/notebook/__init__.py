import pickle

import cfg4py
import matplotlib.pyplot as plt
import numpy as np
import omicron
import pandas as pd
from IPython.display import clear_output
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security

from alpha.config import get_config_dir
from alpha.core.features import fillna, moving_average, predict_by_moving_average
from alpha.core.morph import MorphFeatures
from alpha.plotting import draw_trendline
from alpha.plotting.candlestick import Candlestick


async def init_notebook():
    cfg4py.init(get_config_dir())
    await omicron.init()
    clear_output()