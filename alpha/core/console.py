#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:

请在jupyter lab中调用本文件中的函数。部分函数需要jupyter lab支持

"""
import asyncio
import datetime
import json
import logging
import pprint
import uuid
from typing import Union, List

import aiohttp
import arrow
import cfg4py
import numpy as np
import omicron
import pandas as pd
from IPython.display import Audio, display, clear_output
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.dal import cache
from omicron.models.security import Security
from pandas import DataFrame
from pyemit import emit
from termcolor import colored

from alpha.config import get_config_dir
from alpha.notify.itek import ItekClient
from alpha.plots import create_plot

logger = logging.getLogger(__name__)

cfg = cfg4py.get_instance()
cfg4py.init(get_config_dir())

event_loop = asyncio.get_running_loop()
sched = AsyncIOScheduler(timezone=cfg.tz, loop=event_loop)

sched.start()

# global variables
queued_sound = 0
fired_signals = {}


async def dec_queued_sound():
    global queued_sound
    queued_sound -= 1
    clear_output(wait=True)


async def init():
    await omicron.init()
    # noinspection PyBroadException
    try:
        await emit.stop()
    except Exception:
        pass
    await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn)


async def self_test():
    await read_msg("系统自检消息。通知系统正常工作中。")


async def read_msg(text: str):
    global queued_sound
    global sched

    itek = ItekClient('/notebooks/msg/')
    filename = f"{uuid.uuid4().hex}.mp3"
    sound_file = await itek.tts(text, filename)

    queued_sound += 1
    next_run_time = arrow.now(cfg.tz).datetime + datetime.timedelta(
            seconds=queued_sound * 7)
    sched.add_job(dec_queued_sound, 'date', next_run_time=next_run_time)
    await asyncio.sleep((queued_sound - 1) * 5)
    # noinspection PyTypeChecker
    display(Audio(url=sound_file, autoplay=True))


async def read_fired_signal_msg(msg: dict):
    plot, flag, code = msg.get("plot"), msg.get("flag"), msg.get("code")
    plot_name = msg.get("name")
    sec = Security(code)

    if flag == "long":
        sig_name = "发出买入信号"
    elif flag == "short":
        sig_name = "发出卖出信号"
    else:
        sig_name = "触发"

    text = f"{plot_name}监控策略在{sec.display_name}上{sig_name}"
    msg.update({"股票名": sec.display_name})

    await read_msg(text)


async def read_enter_pool_msg(msg: dict):
    plot_display_name = msg.get("plot_name")
    name = Security(msg.get("code")).display_name
    text = f"{name}进入{plot_display_name}股票池"
    await read_msg(text)


async def on_sig_trade(msg: dict):
    global fired_signals

    plot = msg.get("plot")
    recs = fired_signals.get(plot, [])
    recs.append(msg)
    fired_signals[plot] = recs

    clear_output()
    for k, v in fired_signals:
        df = DataFrame(v)
        df.drop("plot")
        df.set_index("fire_on", inplace=True)
        display(df)


async def on_enter_pool(msg: dict):
    global fired_signals

    plot = msg.get("plot")
    recs = fired_signals.get(plot, [])
    recs.append(msg)
    recs[plot] = recs

    clear_output()
    for k, v in fired_signals:
        df = DataFrame(v)
        df.drop("plot")
        df.set_index("frame", inplace=True)
        display(df)


async def list_momentum_pool(day_offset: int = 1, sort_by='y'):
    start = tf.day_shift(arrow.now().date(), -day_offset)
    key = f"plots.momentum.pool"
    recs = await cache.sys.hgetall(key)
    data = []
    for k, v in recs.items():
        frame, code = k.split(":")
        if arrow.get(frame) < arrow.get(start):
            continue

        sec = Security(code)
        v = json.loads(v)

        frame_type = FrameType(v.get("frame_type"))
        fired = tf.int2time(frame) if frame_type in tf.minute_level_frames else \
            tf.int2date(frame)
        data.append({
            "name":  sec.display_name,
            "code":  code,
            "fired": fired,
            "frame": frame_type.value,
            "y":     round(v.get("y"), 2),
            "vx":    round(v.get("vx"), 1),
            "a":     round(v.get("a"), 4),
            "b":     round(v.get("b"), 4),
            "err":   round(v.get("err"), 4)
        })

    if len(data) == 0:
        print("no data")
    else:
        df = DataFrame(data)
        df.set_index('fired', inplace=True)
        display(df.sort_values(sort_by))


async def visual_mom(code: str, frame: str = None, frame_type='1d'):
    frame = frame or arrow.now(tz=cfg.tz)
    mom = create_plot('momentum')
    await mom.visualize(code, frame, frame_type)


async def copy_mom(code: str, frame: str = None, frame_type: str = '1d', wins=None):
    frame = frame or arrow.now(tz=cfg.tz)
    mom = create_plot('momentum')
    wins = wins or [5, 10, 20]
    features = await mom.copy(code, frame_type, frame, wins)
    for win, feat in zip(wins, features):
        print(f"ma{win}特征:")
        print(round(feat[0], 4), np.round(feat[1], 4), round(feat[2], 1))


async def request(cat, cmd, **kwargs):
    url = f"{cfg.alpha.urls.service}/{cat}/{cmd}"

    async with aiohttp.ClientSession() as client:
        try:
            async with client.post(url, json=kwargs) as resp:
                if resp.status != 200:
                    desc = await resp.content.read()
                    print(colored(f'failed to execute {cmd}:{desc}', 'red'))
                else:
                    return await resp.json()
        except Exception as e:
            print(e)


async def list_monitors(plot: str = None, code: str = None):
    monitors = {}
    for item in await request("monitor", "list", plot=plot, code=code):
        code = item[1].get("code")
        item[1]['name'] = Security(code).display_name
        item[1]['triggers'] = item[2]

        recs = monitors.get(item[0], [])
        recs.append(item[1])
        monitors[item[0]] = recs

    pd.set_option('max_rows', 100)
    if plot is not None:
        plots = [plot]
    else:
        plots = monitors.keys()

    for plot in plots:
        recs = monitors.get(plot)
        df = DataFrame(recs,
                       columns=["name", "code", "frame_type", "flag", "win",
                                "triggers"])
        df.columns = ["name", "code", "frame", "flag", "win", "trigger"]
        print(plot)
        display(df)


async def disable_monitors(code: Union[str, List[str]], frame: str, flag: str,
                           win: int):
    pass


async def add_monitor(code, trigger: str, plot: str, flag: str, frame: str, win: int,
                      hash_keys: tuple = None):
    hash_keys = ("plot", "code", "frame_type", "flag", "win")
    resp = await request('monitor', 'add', code=code, trigger=trigger, plot=plot,
                         flag=flag, frame_type=frame, win=win, hash_keys=hash_keys)
    print(f"{resp} 已加入监控")


async def remove_monitor(job_name=None, code=None, plot=None, frame=None, flag=None):
    if not any([code, plot, frame, flag]):
        resp = await request('monitor', 'remove', all=True)
        print(resp)
        return

    if isinstance(code, list):
        for _code in code:
            resp = await request('monitor', 'remove', job_name=job_name, plot=plot,
                                 code=_code,
                                 frame_type=frame, flag=flag)
            print(resp)
    else:
        resp = await request('monitor', 'remove', job_name=job_name, plot=plot,
                             code=code,
                             frame_type=frame, flag=flag)
        pprint.pprint(resp)


async def list_stock_pool(plot=None, time_offset: int = 3):
    now = arrow.now().date()
    start = tf.day_shift(now, -time_offset)
    if plot is None:
        keys = await cache.sys.keys("plots.*.pool")
    else:
        keys = [f"plots.{plot}.pool"]

    results = []
    for key in keys:
        recs = await cache.sys.hgetall(key)
        data = []
        for k, v in recs.items():
            _frame, code = k.split(":")
            if len(_frame) == 8:
                frame = tf.int2date(int(_frame))
            else:
                frame = tf.int2time(int(_frame))

            if arrow.get(frame) < arrow.get(start):
                continue

            sec = Security(code)
            row = {
                "name":  sec.display_name,
                "code":  code,
                "frame": frame
            }
            row.update(json.loads(v))
            data.append(row)

        print(f"----------{key.lower()}----------")
        df = DataFrame(data=data)
        df.set_index('frame', inplace=True)
        display(df)

        results.append(df)

    return results


async def list_jobs():
    resp = await request('jobs', 'list')
    for job in resp:
        print(job)
