#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging
import pickle

import cfg4py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from omicron.core.frametrigger import FrameTrigger
from omicron.core.tradetimeintervaltrigger import TradeTimeIntervalTrigger
from omicron.core.types import FrameType
from omicron.dal import cache

from alpha.monitor.monitors import create_monitor

logger = logging.getLogger(__name__)

cfg = cfg4py.get_instance()


class Monitor:
    """
    对证券市场而言，一般监控只应该发生在交易时间，本Monitor通过设定interval类别的trigger,
    每个交易自动重安排各项任务来实现此一目的。

    一个进程仅有一个monitor；monitor在执行监控时，将根据需要创建plot对象来完成状态评估。
    """
    monitor_key = "monitor.monitors"

    def __init__(self):
        self.watch_list = {}
        self.sched = None

    def init(self, scheduler=None):
        self.sched = scheduler or AsyncIOScheduler(timezone=cfg.tz)
        self.sched.add_job(self.resume_monitors, 'date')
        if not self.sched.running:
            self.sched.start()

    def _add_watch(self, name: str, job_name: str, trigger: str, **func_args):
        """

        Args:
            name: the name of the monitor
            job_name: the id of the job
            trigger: 如'interval:3' or 'frame:30m:-3'
            **func_args:

        Returns:

        """
        mon = create_monitor(name)

        if trigger.startswith('interval'):
            interval = int(trigger.split(":")[1])
            _trigger = TradeTimeIntervalTrigger(interval)
        elif trigger.startswith('frame'):
            try:
                frame_type, jitter = trigger.split(":")[1:]
                frame_type = FrameType(frame_type)
                jitter = int(jitter)
            except ValueError:
                raise ValueError("format for trigger should like:'frame:30m:5'")
            _trigger = FrameTrigger(frame_type, jitter)
        else:
            raise ValueError(f"trigger type {trigger} not supported")
        self.sched.add_job(mon.evaluate, trigger=_trigger, name=job_name,
                           kwargs=func_args)

    async def watch(self, name: str, trigger: str, **kwargs):
        """
        (plot, code, frame_type, flag)构成惟一的一个监控任务。
        Args:
            name:
            trigger: format like 'interval:3' or 'frame:30m:-3'
            **kwargs:

        Returns:

        """
        persist_data = {
            "jobinfo": {
                "trigger":        trigger,
                "monitor":        name,
            },
            "kwargs":  kwargs
        }

        code = kwargs.get("code")
        frame_type = kwargs.get("frame_type")
        flag = kwargs.get("flag")

        if not all([code, frame_type, flag, name]):
            raise ValueError("missing code, frame_type, flag or plot")

        job_name = ":".join((name, code, frame_type, flag))

        self.watch_list[job_name] = persist_data
        self._add_watch(name, job_name, trigger, **kwargs)
        await cache.sys.hset(self.monitor_key, job_name, pickle.dumps(persist_data,
                                                                      protocol=0))

    async def add_batch(self, name: str, trigger: str, **kwargs):
        """
        todo: 报告错误的参数，比如股票代码。
        todo: 回显已加的监控，含已存在的。
        Args:
            name:
            trigger:
            **kwargs:

        Returns:

        """
        codes = kwargs.get("code_list")
        params = kwargs
        del params["code_list"]
        for code in codes.split(","):
            params['code'] = code
            await self.watch(name, trigger, **params)

    async def resume_monitors(self):
        """
        resume monitors from database, in case of the process is restarted
        Returns:

        """
        logger.info("loading monitor...")
        recs = await cache.sys.hgetall(self.monitor_key)
        for job_name, params in recs.items():
            name, code, frame_type, flag = job_name.split(":")
            params = pickle.loads(params.encode('utf-8'))
            trigger = params.get("jobinfo").get("trigger")
            kwargs = params.get("kwargs")
            self._add_watch(name, job_name, trigger, **kwargs)
            self.watch_list[job_name] = params
        logger.info("done with %s monitor loaded", len(self.watch_list))

        return self.watch_list

    async def remove(self, name: str, code: str, frame_type: str=None, flag: str=None,
                     remove_all=False):
        if remove_all:
            self.sched.remove_all_jobs()
            await cache.sys.delete(self.monitor_key)
            self.watch_list = {}
        else:
            if all((name, code, frame_type, flag)):
                job_name = ":".join((name, code, frame_type, flag))
                for job in self.sched.get_jobs():
                    if job.name == job_name:
                        self.sched.remove_job(job.id)
                        await cache.sys.hdel(self.monitor_key, job_name)
                        del self.watch_list[job.name]
            elif all((name, code)):
                for job in self.sched.get_jobs():
                    if job.name.startswith(":".join((name,code))):
                        self.sched.remove_job(job.id)
                        await cache.sys.hdel(self.monitor_key, job.name)
                        del self.watch_list[job.name]
            elif any((name, code)):
                for job in self.sched.get_jobs():
                    if job.name.find(f"{name}:") != -1 or job.name.find(f"{code}:") \
                            != -1:
                        self.sched.remove_job(job.id)
                        await cache.sys.hdel(self.monitor_key, job.name)
                        del self.watch_list[job.name]
            else:
                raise ValueError("either name or code must present.")

    async def list_watch(self, code: str = '', frame_type: str = '', plot: str = '',
                         flag: str = ''):
        heap = self.watch_list.keys()
        for _filter in [code, frame_type, plot, flag]:
            if _filter:
                tmp_heap = []
                for watch in heap:
                    if watch.find(_filter) != -1:
                        tmp_heap.append(watch)
                heap = tmp_heap

        results = []
        for key in heap:
            plot, code, frame_type, flag = key.split(":")
            params = self.watch_list[key].get("kwargs")
            trigger = self.watch_list[key].get("jobinfo").get("trigger")
            results.append([plot, params, trigger])
        return results


monitor = Monitor()

__all__ = ['monitor']
