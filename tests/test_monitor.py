import unittest
from collections import namedtuple
from unittest.mock import patch, MagicMock

import cfg4py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from omicron.core.lang import async_run

from alpha.core.monitors import MonitorManager
from tests.base import AbstractTestCase

cfg = cfg4py.get_instance()


class MyTestCase(AbstractTestCase):
    @async_run
    async def test_watch(self):
        sched = AsyncIOScheduler(timezone=cfg.tz)
        monitor = MonitorManager()
        monitor.init(sched)

        plot = 'momentum'
        freq = 3
        trade_time_only = True

        code = '000001.XSHE'
        flag = 'both'
        frame_type = '1d'
        mom = 0.01

        with patch('omicron.core.timeframe.tf.is_trade_day', side_effect=[True]):
            await monitor.watch(plot, freq=freq, trade_time_only=trade_time_only,
                                code=code, flag=flag, mom=mom, frame_type=frame_type)
            job_id = ":".join((plot, code, frame_type, flag))
            self.assertEqual(3, len(sched.get_jobs()))
            recs = await monitor.resume_monitors()
            self.assertDictEqual({job_id: {
                "jobinfo": {
                    "freq":            3,
                    "plot":            plot,
                    "trade_time_only": trade_time_only
                },
                "kwargs":  {
                    "mom":        mom,
                    'code':       code,
                    'flag':       flag,
                    'frame_type': frame_type
                }
            }}, recs)
            await monitor.remove(plot, code, frame_type, flag)
            self.assertEqual(1, len(sched.get_jobs()))

    def test_find_job(self):
        plot, code, frame_type, flag, win = "maline:000001.XSHG:1d:both:5".split(":")

        from alpha.core.monitors import mm
        mm.sched = MagicMock()

        def _get_jobs():
            class _:
                name = "maline:000001.XSHG:1d:both:5"

            return namedtuple()

        mm.sched.get_jobs = _get_jobs()
        job = mm.find_job(plot, code, flag, win, frame_type)
        print(job)


if __name__ == '__main__':
    unittest.main()
