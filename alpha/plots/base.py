#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pyemit import emit

logger = logging.getLogger(__name__)

class AbstractPlot():
    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone='Asia/Shanghai')
        self.stock_pool = {}

    async def _start(self):
        await emit.start(emit.Engine.REDIS, start_server=False)

        self.stock_pool = await self.load_watch_list()

    async def start(self):
        raise NotImplemented("subclass must implement this")

    async def load_watch_list(self):
        pass

    async def on_new_data(self):
        pass