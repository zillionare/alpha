import logging
import unittest

import cfg4py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from omicron.core.lang import async_run
from omicron.core.types import FrameType
from omicron.models.securities import Securities

from alpha.core.monitors import mm
from alpha.plots.maline import MaLine
from tests.base import AbstractTestCase

cfg = cfg4py.get_instance()

logger = logging.getLogger(__name__)


class MyTestCase(AbstractTestCase):

    @async_run
    async def test_evaluate(self):
        from pyemit import emit
        await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn)
        scheduler = AsyncIOScheduler(timezone=cfg.tz)
        mm.init(scheduler)
        plot = MaLine()
        plot.max_fire_time = 2
        secs = Securities().choose(['stock'])
        for code in secs:
            await plot.evaluate(code, FrameType.DAY, 'both', 5,
                                job_name='mock_job_name')
            await plot.evaluate(code, FrameType.DAY, 'both', 5,
                                job_name='mock_job_name')


if __name__ == '__main__':
    unittest.main()
