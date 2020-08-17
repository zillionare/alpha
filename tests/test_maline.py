import asyncio
import logging
import unittest

import arrow
import cfg4py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from omicron.core.lang import async_run
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.securities import Securities

from alpha.plots.maline import MALinePlot
from tests.base import AbstractTestCase

cfg = cfg4py.get_instance()

logger = logging.getLogger(__name__)


class MyTestCase(AbstractTestCase):
    @async_run
    async def test_watch(self):
        scheduler = AsyncIOScheduler(timezone=cfg.tz)
        plot = MALinePlot(scheduler)

        plot.watch(freq=0.02, code='000001', frame_type=FrameType.DAY, ma_win=[5, 10])
        plot.watch(freq=0.02, code='000002', frame_type=FrameType.MIN1, ma_win=[5, 10])
        await asyncio.sleep(3)

    @async_run
    async def test_evaluate(self):
        from pyemit import emit
        await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn)
        scheduler = AsyncIOScheduler(timezone=cfg.tz)
        plot = MALinePlot(scheduler)
        secs = Securities().choose(['stock'])
        for code in secs:
            dt = tf.floor(arrow.get('2020-8-6'), FrameType.DAY)
            await plot.evaluate(code, [60], FrameType.DAY, dt=dt)


if __name__ == '__main__':
    unittest.main()
