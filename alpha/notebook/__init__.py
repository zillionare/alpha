from typing import Union
import datetime
import pickle

import arrow
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
from alpha.core.features import *
from alpha.core.morph import MorphFeatures
from alpha.features.volume import top_volume_direction
from alpha.plotting import draw_trendline
from alpha.plotting.candlestick import Candlestick
from pretty_html_table import build_table
from alpha.core.notify import send_html_email, say
import os
from jqdatasdk import *
import jqdatasdk as jq
from arrow import Arrow

g = {

}

async def init_notebook(use_omicron=True):
    cfg4py.init(get_config_dir())
    if use_omicron:
        await omicron.init()
    else:
        init_jq()

    clear_output()


def init_jq():
    account = os.getenv("JQ_ACCOUNT")
    password = os.getenv("JQ_PASSWORD")

    jq.auth(account, password)
    g["valuations"] = jq.get_fundamentals(query(valuation))
    g["secs"] = jq.get_all_securities()

def jq_get_name(code):
    secs = g["secs"]
    return secs[secs.index==code].iloc[0]["display_name"]

def jq_get_valuation(code):
    valuations = g["valuations"]

    return int(valuations[valuations.code==code].iloc[0]["market_cap"])

def jq_choose_stocks(exclude_st=True, exclude_688=True, valuation_range=(100,2000)):
    result = g["secs"]
    if exclude_688:
        result = result[result.index.str.startswith("688")==False]
    if exclude_st:
        result = result[result.display_name.str.find("ST") == -1]

    codes = []
    for code in result.index:
        try:
            if valuation_range[0] <= jq_get_valuation(code) <= valuation_range[1]:
                codes.append(code)
        except Exception:
            pass

    return codes

def jq_get_bars(code:str, n:int, frame_type:str='1d', end:Union[str, datetime.date, datetime.datetime, Arrow]=None):
    fields = ['date', 'open', 'high', 'low', 'close', 'volume']
    if frame_type in ['1d', '1w', '1M', '1Y']:
        end = arrow.get(end).date() if end else arrow.now().date()
    else:
        end = arrow.get(end).datetime if end else arrow.now().datetime

    bars = jq.get_bars(code, n, frame_type, end_dt = end, fields = fields, df=False)
    bars.dtype.names = ['frame', 'open', 'high', 'low', 'close', 'volume']
    return bars

__all__ = [
    "plt",
    "np",
    "omicron",
    "pd",
    "tf",
    "FrameType",
    "Securities",
    "Security",
    "Candlestick",
    "fillna",
    "draw_trendline",
    "predict_by_moving_average",
    "moving_average",
    "polyfit",
    "init_notebook",
    "MorphFeatures",
    "top_volume_direction",
    "arrow",
    "pickle",
    "clear_output",
    "send_html_email",
    "build_table",
    "say",
    "jq_get_name",
    "jq_get_valuation",
    "jq_choose_stocks",
    "jq_get_bars"
]
