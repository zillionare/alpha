from typing import Union
import datetime
import time

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
from alpha.plotting import draw_trendline, draw_ma_lines
from alpha.plotting.candlestick import Candlestick
from pretty_html_table import build_table
from alpha.core.notify import send_html_email, say
from alpha.core.rsi_stats import RsiStats
import os
from jqdatasdk import *
import jqdatasdk as jq
from arrow import Arrow
from alpha.utils import *

g = {}


async def init_notebook(adaptor='both'):
    cfg4py.init(get_config_dir())
    if adaptor in ['omicron', 'both']:
        await omicron.init()
    if adaptor in ['jq', 'both']:
        init_jq()

    clear_output()

def init_jq():
    account = os.getenv("JQ_ACCOUNT")
    password = os.getenv("JQ_PASSWORD")

    jq.auth(account, password)
    g["valuations"] = jq.get_fundamentals(query(valuation))
    g["secs"] = jq.get_all_securities()


def jq_get_code(name):
    secs = g["secs"]
    return secs[secs.display_name == name].iloc[0].index


def jq_get_name(code):
    secs = g["secs"]
    return secs[secs.index == code].iloc[0]["display_name"]


def jq_get_ipo_date(code):
    secs = g["secs"]
    return secs[secs.index == code].iloc[0]["start_date"].date()


def jq_get_market_cap(code):
    """获取`code`最新的总市值，以亿为单位"""
    valuations = g["valuations"]

    return valuations[valuations.code == code].iloc[0]["market_cap"]


def jq_get_circulating_market_cap(code):
    """获取`code`最新的流通市值，以亿为单位"""
    valuations = g["valuations"]

    return valuations[valuations.code == code].iloc[0]["circulating_market_cap"]


def jq_get_turnover(code):
    """获取`code`最新的换手率"""
    valuations = g["valuations"]

    return valuations[valuations.code == code].iloc[0]["turnover_ratio"]


def jq_choose_stocks(exclude_st=True, exclude_688=True, valuation_range=(100, 2000)):
    result = g["secs"]
    if exclude_688:
        result = result[result.index.str.startswith("688") == False]
    if exclude_st:
        result = result[result.display_name.str.find("ST") == -1]

    codes = []
    for code in result.index:
        try:
            if valuation_range[0] <= jq_get_market_cap(code) <= valuation_range[1]:
                codes.append(code)
        except Exception:
            pass

    return codes


def jq_get_bars(
    code: str,
    n: int,
    frame_type: str = "1d",
    end: Union[str, datetime.date, datetime.datetime, Arrow] = None,
):
    fields = ["date", "open", "high", "low", "close", "volume"]
    end = arrow.get(end).datetime if end else arrow.now().datetime

    bars = jq.get_bars(
        code, n, frame_type, end_dt=end, fields=fields, df=False, include_now=True
    )
    bars.dtype.names = ["frame", "open", "high", "low", "close", "volume"]
    return bars

async def get_bars(code:str, n:int, frame_type:str='1d', end: Frame=None):
    """获取`code`的`n`个`frame_type`的K线数据。

    在notebook中使用的辅助函数，简化为一些常用操作。

    Args:
        code ([type]): [description]
        n ([type]): [description]
        frame_type ([type], optional): [description]. Defaults to '1d'.
        end ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if not "." in code:
        if code.startswith("6"):
            code += ".XSHG"
        else:
            code += ".XSHE"

    ft = FrameType(frame_type)

    end = arrow.get(end) or arrow.now()

    if ft in tf.minute_level_frames:
        start = tf.shift(tf.floor(end, ft), -n + 1, ft)
    else:
        start = tf.shift(end, -n + 1, ft)

    sec = Security(code)
    try:
        bars = await sec.load_bars(start, end, ft)
    except Exception as e:
        return None
        
    return bars[np.isfinite(bars["close"])]

def jq_get_turnover_realtime(code, volume, close):
    """获取`code`最新的换手率。 jq_get_turnover无法获取盘中最新的换手率数据。该数据只能在24：00以后才能获取。

    Args:
        code ([type]): [description]
        volume ([type]): [description]
        close ([type]): [description]

    Returns:
        [type]: [description]
    """
    return volume * close / (1e8 * jq_get_circulating_market_cap(code))


def mail_notify(subject: str, model: str, params: dict, report: pd.DataFrame):
    """key must be unique, contains a date"""

    html = f"""
    <html>
    <head>
        <style>
            table, th, td {{
                border-collapse: collapse;
                border: 0px;
                color: #666;
            }}
        </style>
    </head>
    <body>
    <h2 style="color:#c21">{model}</h2>
    {build_table(report, "grey_light")}

    <p><strong>Parameters:</strong></p>
    <p>{" ".join([f"{k}:{v}" for k,v in params.items()])}
    </body>
    </html>
    """
    send_html_email(subject, html)

async def scan(trigger, n, frame_type=FrameType.DAY, tm=None, test=False):
    results = []

    end = arrow.get(tm) if tm else arrow.now()
    start = tf.shift(tf.floor(end, frame_type), -n+1, frame_type)
    codes =Securities().choose(["stock"])

    t0 = time.time()
    for i, code in enumerate(codes):
        if test and i >= 10:
            break
        if (i+1) % 500 == 0:
            elapsed = int(time.time() - t0)
            eta = int((len(codes) - i) * elapsed / (i+1))
            print(f"progress: {i + 1}/{len(codes)}, elapsed: {elapsed}, ETA: {eta}")
        sec = Security(code)
        name = sec.display_name
        try:
            bars = await sec.load_bars(start, end, frame_type)
            bars = bars[np.isfinite(bars["close"])]

            trigger(code, name, bars, results, frame_type)
        except Exception:
            continue

    say("扫描结束")
    return results
