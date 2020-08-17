import unittest
from unittest.mock import Mock, patch

import cfg4py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from omicron.core.lang import async_run
from omicron.core.types import FrameType

from alpha.core.monitor import Monitor
from tests.base import AbstractTestCase

cfg = cfg4py.get_instance()


class MyTestCase(AbstractTestCase):
    @async_run
    async def test_watch(self):
        sched = AsyncIOScheduler(timezone=cfg.tz)
        monitor = Monitor()
        monitor.init(sched)

        plot = 'momentum'
        freq=3
        trade_time_only = True

        code = '000001.XSHE'
        flag = 'both'
        frame_type='1d'
        mom = 0.01

        with patch('omicron.core.timeframe.tf.is_trade_day', side_effect=[True]):
            await monitor.watch(plot, freq=freq, trade_time_only=trade_time_only,
                                code=code, flag=flag, mom=mom,frame_type=frame_type)
            job_id = ":".join((plot,code,frame_type,flag))
            self.assertEqual(3,len(sched.get_jobs()))
            recs = await monitor.load_watch_list()
            self.assertDictEqual({job_id: {
                "jobinfo": {
                    "freq": 3,
                    "plot": plot,
                    "trade_time_only": trade_time_only
                },
                "kwargs":  {
                    "mom": mom,
                    'code':       code,
                    'flag':       flag,
                    'frame_type': frame_type
                }
            }}, recs)
            await monitor.remove(plot, code, frame_type, flag)
            self.assertEqual(1, len(sched.get_jobs()))


if __name__ == '__main__':
    unittest.main()
