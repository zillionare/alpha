#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import json
import logging
from typing import Any

import arrow
import cfg4py
from alpha.core.monitors import mm
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.dal import cache
from omicron.models.security import Security
from pyemit import emit

from alpha.core.enums import Events

cfg = cfg4py.get_instance()
logger = logging.getLogger(__name__)


class BasePlot:
    def __init__(self, display_name: str):
        self.name = self.__class__.__name__.lower()
        self.display_name = display_name
        self.baselines = {}
        self.memory = {}

    def set_baseline(self, key:str, value:Any):
        self.baselines[key] = value

    def baseline(self, key):
        return self.baselines.get(key)

    def parse_monitor_settings(self, **kwargs):
        raise NotImplementedError("subclass must implement this")

    async def get_bars(self, code: str, n: int, frame_type: FrameType,
                       end_dt: Frame = None):
        end_dt = end_dt or arrow.now(tz=cfg.tz)

        sec = Security(code)
        start = tf.shift(tf.floor(end_dt, frame_type), -n + 1, frame_type)
        return await sec.load_bars(start, end_dt, frame_type)

    async def copy(self, *args, **kwargs):
        pass

    async def evaluate(self, code:str, **params):
        pass

    async def scan(self, end: Frame, frame_type: FrameType, codes=None):
        raise NotImplementedError("subclass must implement this")

    def remember(self, code: str, frame_type: FrameType, key: str, value: Any):
        item = self.memory.get(f"{code}:{frame_type.value}", {})
        item[key] = value
        self.memory[f"{code}:{frame_type.value}"] = item

    def recall(self, code: str, frame_type: FrameType, key: str):
        return self.memory.get(f"{code}:{frame_type.value}", {}).get(key)

    async def enter_stock_pool(self, code, frame, frame_type: FrameType, **kwargs):
        if frame_type in tf.day_level_frames:
            iframe = tf.date2int(frame)
        else:
            iframe = tf.time2int(frame)

        kwargs.update({"frame_type": frame_type.value})
        await cache.sys.hset(f"plots.{self.name}.pool",
                             f"{iframe}:{code}",
                             json.dumps(kwargs))

        await mm.evaluate('momentum', {
            "name": 'frame',
            "frame_type": '30m'
        },
                          code=code,
                          frame_type=frame_type.value,
                          flag="both", win=5)

        kwargs.update({
            "code":      code,
            "frame":     iframe,
            "plot":      self.name,
            "plot_name": self.display_name
        })
        await emit.emit(Events.plot_pool, kwargs)

    async def fire_trade_signal(self, flag: str, code: str, fire_on: Frame,
                                frame_type: FrameType,
                                **kwargs):
        convert = tf.date2int if frame_type in tf.day_level_frames else tf.time2int
        fire_on = convert(fire_on)

        self.remember(code, frame_type, "trend", flag)
        sec = Security(code)

        event = {
            "sec":        sec.display_name,
            "plot":       self.name,
            "name":       self.display_name,
            "code":       code,
            "flag":       flag,
            "fire_on":    fire_on,
            "frame_type": frame_type.value
        }
        event.update(kwargs)

        logger.info("%s", event.values())
        await emit.emit(Events.sig_trade, kwargs)
        d = {
            f"{code}:{fire_on}": json.dumps(kwargs)
        }
        await cache.sys.hmset_dict(f"plots.{self.name}.fired", d)
