#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import json
import logging

import cfg4py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.dal import cache
from omicron.models.security import Security
from pyemit import emit

from alpha.core.enums import Events
from alpha.core.monitor import Monitor

cfg = cfg4py.get_instance()
logger = logging.getLogger(__name__)


class AbstractPlot:
    def __init__(self):
        self.name = self.__class__.__name__

    async def evaluate(self, *args, **kwargs):
        raise NotImplementedError("subclass must implement this")

    async def get_bars(self, code: str, n: int, frame_type: FrameType, end_dt: Frame):
        sec = Security(code)
        start = tf.shift(tf.floor(end_dt, frame_type), -n + 1, frame_type)
        return await sec.load_bars(start, end_dt, frame_type)

    async def copy(self, *args, **kwargs):
        pass

    async def fire(self, flag: str, code: str, fire_on: Frame, frame_type:FrameType,
                   **kwargs):
        convert = tf.date2int if frame_type in tf.day_level_frames else tf.time2int
        fire_on = convert(fire_on)
        frame = frame_type.value

        logger.info("%s fired %s on (%s,%s,%s)", self.name, flag, code, fire_on,frame)
        event = Events.sig_long if flag == "long" else "short"
        await emit.emit(event, kwargs)
        d = {
            f"{code}:{fire_on}": json.dumps(kwargs)
        }
        await cache.sys.hmset_dict(f"plots.{self.name}.signals", d)
