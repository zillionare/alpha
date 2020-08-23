#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import json
import logging

import arrow
import cfg4py
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.dal import cache
from omicron.models.security import Security
from pyemit import emit

from alpha.core.enums import Events

cfg = cfg4py.get_instance()
logger = logging.getLogger(__name__)


class BasePlot:
    def __init__(self):
        self.name = self.__class__.__name__

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