#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging
import pickle

import arrow
import cfg4py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from omicron.core.timeframe import tf
from omicron.dal import cache

logger = logging.getLogger(__name__)

cfg = cfg4py.get_instance()


class Monitor:
    """
    对证券市场而言，一般监控只应该发生在交易时间，本Monitor通过设定interval类别的trigger,
    每个交易自动重安排各项任务来实现此一目的。

    一个进程仅有一个monitor；monitor在执行监控时，将根据需要创建plot对象来完成状态评估。
    """
    monitor_key = "plots.monitor"

    def __init__(self):
        self.watch_list = {}

    def init(self, scheduler=None):
        self.sched = scheduler or AsyncIOScheduler(timezone=cfg.tz)
        self.sched.add_job(self.reset_jobs, 'cron', hour=2)
        self.sched.add_job(self.load_watch_list, 'date')
        if not self.sched.running:
            self.sched.start()

    def create_plot(self, name):
        if name == 'momentum':
            from alpha.plots.momentum import Momentum
            return Momentum()

        raise ValueError(f"{name} is not supported yet")

    def reset_jobs(self):
        self.sched.remove_all_jobs()
        for job_name, params in self.watch_list.items():
            freq = params.get("jobinfo").get("freq")
            plot = params.get("jobinfo").get("plot")
            trade_time_only = params.get("jobinfo").get("trade_time_only")
            self._add_watch(plot, job_name, freq, trade_time_only,
                            **params.get("kwargs"))

    def _add_watch(self, plot_name: str, jobid: str, freq: int, trade_time_only=True,
                   **kwargs):
        plot = self.create_plot(plot_name)
        now = arrow.now(cfg.tz).floor('hour')
        if trade_time_only:
            if not tf.is_trade_day(now):
                logger.info("No monitor job is scheduled as today is not trading day")
                return

            # apscheduler bug: object has no attribute '_utcoffset'
            start1 = now.replace(hour=9, minute=30).strftime("%Y-%m-%d %H:%M:%S")
            end1 = now.replace(hour=11, minute=30).strftime("%Y-%m-%d %H:%M:%S")

            start2 = now.replace(hour=13).strftime("%Y-%m-%d %H:%M:%S")
            end2 = now.replace(hour=15).strftime("%Y-%m-%d %H:%M:%S")

            self.sched.add_job(plot.evaluate, 'interval', minutes=freq,
                               start_date=start1, end_date=end1, kwargs=kwargs,
                               timezone=cfg.tz, name=jobid)
            self.sched.add_job(plot.evaluate, 'interval', minutes=freq,
                               start_date=start2, end_date=end2, kwargs=kwargs,
                               timezone=cfg.tz, name=jobid)
        else:
            self.sched.add_job(plot.evaluate, 'interval', minutes=freq,
                               kwargs=kwargs, name=jobid)

    async def watch(self, plot: str, freq: int = 3, trade_time_only=True, **kwargs):
        """
        (code, frame_type, flag)构成惟一的一个监控任务。
        Args:
            code:
            frame_type:
            flag:
            freq:
            **kwargs:

        Returns:

        """
        persist_data = {
            "jobinfo": {
                "freq":            freq,
                "trade_time_only": trade_time_only,
                "plot":            plot
            },
            "kwargs":  kwargs
        }

        code = kwargs.get("code")
        frame_type = kwargs.get("frame_type")
        flag = kwargs.get("flag")

        if not all([code, frame_type, flag, plot]):
            raise ValueError("missing code, frame_type, flag or plot")

        job_name = ":".join((plot, code, frame_type, flag))

        self.watch_list[job_name] = persist_data
        self._add_watch(plot, job_name, freq, trade_time_only, **kwargs)
        await cache.sys.hset(self.monitor_key, job_name, pickle.dumps(persist_data,
                                                                      protocol=0))

    async def load_watch_list(self):
        logger.info("loading monitors...")
        recs = await cache.sys.hgetall(self.monitor_key)
        for job_name, params in recs.items():
            self.watch_list[job_name] = pickle.loads(params.encode('utf-8'))
        logger.info("done with %s monitors loaded", len(self.watch_list))
        return self.watch_list

    async def remove(self, plot: str, code: str, frame_type: str, flag: str):
        job_name = ":".join((plot, code, frame_type, flag))
        for job in self.sched.get_jobs():
            if job.name == job_name:
                self.sched.remove_job(job.id)
                await cache.sys.hdel(self.monitor_key, job_name)

    async def list_watch(self, code: str = '', frame_type: str = '', plot: str = '',
                         flag: str = ''):
        heap = self.watch_list.keys()
        for filter in [code, frame_type, plot, flag]:
            if filter:
                tmp_heap = []
                for watch in heap:
                    if watch.find(filter) != -1:
                        tmp_heap.append(watch)
                heap = tmp_heap

        results = [item.split(":") for item in heap]
        return results



monitor = Monitor()

__all__ = ['monitor']
