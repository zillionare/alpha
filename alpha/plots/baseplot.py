#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import json
import logging
from functools import lru_cache
from typing import Any

import arrow
import cfg4py
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.dal import cache
from omicron.models.security import Security
from pyemit import emit

from alpha.core.enums import Events
from alpha.core.monitor import monitor

cfg = cfg4py.get_instance()
logger = logging.getLogger(__name__)


class BasePlot:
    def __init__(self, display_name: str):
        self.name = self.__class__.__name__
        self.display_name = display_name
        self.baselines = {}
        self.memory = {}

    async def evaluate(self, *args, **kwargs):
        raise NotImplementedError("subclass must implement this")

    async def get_bars(self, code: str, n: int, frame_type: FrameType,
                       end_dt: Frame = None):
        end_dt = end_dt or arrow.now(tz=cfg.tz)

        sec = Security(code)
        start = tf.shift(tf.floor(end_dt, frame_type), -n + 1, frame_type)
        return await sec.load_bars(start, end_dt, frame_type)

    async def copy(self, *args, **kwargs):
        pass

    async def pooling(self, end: Frame, frame_type: FrameType, codes=None):
        raise NotImplementedError("subclass must implement this")

    @lru_cache
    def baseline(self, name: str, frame_type: FrameType, ma: str = None):
        if name in self.baselines:
            return self.baselines.get(name).get(frame_type.value)

        return self.baselines.get(ma).get(frame_type.value).get(name)

    def remember(self, code: str, frame_type: FrameType, key: str, value: Any):
        item = self.memory.get(f"{code}:{frame_type.value}", {})
        item[key] = value
        self.memory[f"{code}:{frame_type.value}"] = item

    def recall(self, code: str, frame_type: FrameType, key: str):
        return self.memory.get(f"{code}:{frame_type.value}", {}).get(key)

    async def enter_stock_pool(self, code, frame_type, frame, params):
        if frame_type in tf.day_level_frames:
            iframe = tf.date2int(frame)
        else:
            iframe = tf.time2int(frame)

        await cache.sys.hset("plots.momentum.pool", f"{iframe}:{code}", json.dumps({
            "frame_type": frame_type.value,
            "params":     params
        }))

        await monitor.watch('maline', "interval:1",
                            code=code,
                            frame_type=frame_type.value,
                            flag="long", win=5)

        await emit.emit(Events.plot_pool, {
            "code":       code,
            "name":       Security(code).display_name,
            "frame_type": frame_type.value,
            "frame":      str(frame),
            "plot":       self.name,
            "plot_name":  self.display_name,
            "params":     params
        })

    async def fire_trade_signal(self, flag: str, code: str, fire_on: Frame,
                                frame_type: FrameType,
                                **kwargs):
        convert = tf.date2int if frame_type in tf.day_level_frames else tf.time2int
        fire_on = convert(fire_on)
        frame = frame_type.value

        self.remember(code, frame_type, "trend", flag)
        sec = Security(code)
        logger.info("%s fired %s on (%s,%s,%s)",
                    self.name, flag, sec.display_name, fire_on, frame)

        kwargs.update({
            "monitor":    self.name,
            "name":       self.display_name,
            "code":       code,
            "flag":       flag,
            "fire_on":    fire_on,
            "frame_type": frame_type.value
        })
        await emit.emit(Events.sig_trade, kwargs)
        d = {
            f"{code}:{fire_on}": json.dumps(kwargs)
        }
        await cache.sys.hmset_dict(f"plots.{self.name}.fired", d)
