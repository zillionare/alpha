#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors:

请在jupyter lab中调用本文件中的函数。部分函数需要jupyter lab支持

"""
import json
import logging

logger = logging.getLogger(__name__)

import aiohttp
import pprint
import datetime
from termcolor import colored
import uuid
import asyncio
import arrow
from pandas import DataFrame
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from IPython.display import Audio, display, clear_output

import cfg4py
from pyemit import emit

import omicron
from omicron.dal import cache
from omicron.models.security import Security
from omicron.core.timeframe import tf
from omicron.core.types import FrameType

from alpha.notify.itek import ItekClient
from alpha.core.enums import Events
from alpha.config import get_config_dir
from alpha.plots import create_plot

from alpha import plots

cfg = cfg4py.get_instance()
cfg4py.init(get_config_dir())

sched = AsyncIOScheduler(timezone=cfg.tz)
sched.start()

queued_sound = 0


async def dec_queued_sound():
    global queued_sound
    queued_sound -= 1
    clear_output(wait=True)


async def init():
    await omicron.init()
    emit.register(Events.sig_trade, on_sig_trade)
    emit.register(Events.self_test, self_test)
    emit.register(Events.plot_pool, on_enter_pool)
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


async def on_sig_trade(msg: dict):
    mon, name, flag, code = msg.get("monitor"), msg.get('name'), msg.get(
        "flag"), msg.get("code")
    sec = Security(code)
    if mon and name and flag:
        if flag == 'long':
            await read_msg(f"监控{name}在股票{sec.display_name}发出买入信号")
        else:
            await read_msg(f"监控{name}在股票{sec.display_name}上发出卖出信号")


async def on_enter_pool(msg: dict):
    plot_name = msg.get("plot")
    plot_display_name = msg.get("plot_name")
    name = Security(msg.get("code")).display_name
    text = f"{name}进入{plot_display_name}股票池"
    print(text)
    await read_msg(text)
    plot = plots.create_plot(plot_name)
    del msg['plot']
    del msg['plot_name']
    if hasattr(plot, 'visualize'):
        await plot.visualize(**msg)


async def list_moment_pool(start=None):
    if isinstance(start, str):
        start = arrow.get(start)
    key = f"plots.momentum.pool"
    recs = await cache.sys.hgetall(key)
    data = []
    for k, v in recs.items():
        frame, code = k.split(":")
        if start and arrow.get(frame) < start:
            continue

        sec = Security(code)
        v = json.loads(v)

        frame_type = FrameType(v.get("frame_type"))
        data.append({
            "name":  sec.display_name,
            "code":  code,
            "fired": tf.int2date(
                frame) if frame_type in tf.day_level_frames else tf.int2time(frame),
            "frame": frame_type.value,
            "y":     round(v.get("params").get("y"), 2),
            "vx":    round(v.get("params").get("vx"), 1),
            "a":     round(v.get("params").get("a"), 4),
            "b":     round(v.get("params").get("b"), 4),
            "err":   round(v.get("params").get("err"), 4)
        })

    df = DataFrame(data)
    display(df.sort_values('fired'))


async def visual_mom(code: str, frame: str = None, frame_type='1d'):
    frame = frame or arrow.now(tz=cfg.tz)
    mom = create_plot('momentum')
    await mom.visualize(code, frame, frame_type)


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


async def list_monitors():
    monitors = {}
    for item in await request("monitor", "list"):
        code = item[1].get("code")
        item[1]['name'] = Security(code).display_name
        item[1]['triggers'] = item[2]

        recs = monitors.get(item[0], [])
        recs.append(item[1])
        monitors[item[0]] = recs

    momentum = monitors.get("momentum")
    df = DataFrame(momentum,
                   columns=["name", "code", "frame_type", "flag", "win", "triggers"])
    df.columns = ["name", "code", "frame", "flag", "win", "trigger"]
    print("动量监控")
    display(df)

    maline = monitors.get("maline")
    df = DataFrame(maline,
                   columns=['name', 'code', 'frame_type', 'flag', 'win', 'triggers'])
    df.columns = ["name", "code", "frame", "flag", "win", "trigger"]

    print("均线监控")
    display(df)


async def add_monitor(code, trigger:str, plot:str, flag:str, frame:str, win:int,
                      hash_keys:tuple=None):
    hash_keys = ("plot", "code", "frame_type", "flag", "win")
    resp = await request('monitor', 'add', code=code, trigger=trigger, plot=plot,
                         flag=flag, frame_type=frame, win=win,hash_keys=hash_keys)
    data = []
    for item in resp:
        rec = {"plot": item[0], "trigger": item[2]}
        rec.update(item[1])
        rec['name'] = Security(rec.get('code')).display_name
        data.append(rec)
    df = DataFrame(data, columns=['plot', 'name', 'code', 'flag', 'frame_type', 'win',
                                  'trigger'])
    df.columns = ['plot', 'name', 'code', 'flag', 'frame', 'win', 'trigger']

    display(df)


async def remove_monitor(code=None, plot=None, frame=None, flag=None):
    if not any([code, plot, frame, flag]):
        resp = await request('monitor', 'remove', all=True)
        print(resp)
        return

    if isinstance(code, list):
        for _code in code:
            resp = await request('monitor', 'remove', name=plot, code=_code,
                                 frame_type=frame, flag=flag)
            print(resp)
    else:
        resp = await request('monitor', 'remove', name=plot, code=code,
                             frame_type=frame, flag=flag)
        pprint.pprint(resp)


async def list_stock_pool(plot, start=None):
    if isinstance(start, str):
        start = arrow.get(start)
    key = f"plot.{plot}"
    print(await cache.sys.hgetall(key))


