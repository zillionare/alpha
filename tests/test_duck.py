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

from alpha.plots.duck import DuckPlot
from alpha.plots.maline import MALinePlot
from tests.base import AbstractTestCase

cfg = cfg4py.get_instance()

logger = logging.getLogger(__name__)


class MyTestCase(AbstractTestCase):
    @async_run
    async def test_evaluate(self):
        scheduler = AsyncIOScheduler(timezone=cfg.tz)
        plot = DuckPlot(scheduler)
        secs = Securities().choose(['stock'])
        for code in secs:
            dt = tf.floor(arrow.get('2020-7-24'), FrameType.DAY)
            try:
                await plot.evaluate(code, FrameType.DAY, dt=dt)
            except Exception as e:
                logger.exception(e)


if __name__ == '__main__':
    unittest.main()
