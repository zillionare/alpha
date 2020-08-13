#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging
from typing import Union, List

from omicron.core.types import FrameType
from pyemit import emit

logger = logging.getLogger(__name__)


class Monitor:
    def __init__(self):
        pass

    async def start(self):
        emit.register("alpha/on_new_data", self.on_new_data)
        emit.register("alpha/register_plot", self.register_plot)
        await emit.start(emit.Engine.REDIS, start_server=False)

    async def on_new_data(self, data: dict):
        code = data.get("code")
        bars = data.get("bars")
        frame_type = data.get("frame_type")

        plot = self.find_plot(code, frame_type)
        if plot:
            await plot.on_new_data(code, bars, frame_type)

    def find_plot(self, code, frame_type):
        pass

    def register_plot(self, code:Union[str,List[str]], frame_type: FrameType,
                                plot:object):
        pass

