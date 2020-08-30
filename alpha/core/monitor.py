#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging
import pickle
import re

import cfg4py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from omicron.core.frametrigger import FrameTrigger
from omicron.core.tradetimeintervaltrigger import TradeTimeIntervalTrigger
from omicron.core.types import FrameType
from omicron.dal import cache
from pyemit import emit

from alpha.core.enums import Events
from alpha.plots import create_plot

logger = logging.getLogger(__name__)

cfg = cfg4py.get_instance()


class Monitor:
    """
    对证券市场而言，一般监控只应该发生在交易时间，本Monitor通过设定interval类别的trigger,
    每个交易自动重安排各项任务来实现此一目的。

    一个进程仅有一个monitor；monitor在执行监控时，将根据需要创建plot对象来完成状态评估。
    """
    monitor_key = "monitors"

    def __init__(self):
        self.watch_list = {}
        self.sched = None

    def init(self, scheduler=None):
        self.sched = scheduler or AsyncIOScheduler(timezone=cfg.tz)
        self.sched.add_job(self.resume_monitors, 'date')
        self.sched.add_job(self.self_test, 'cron', day_of_week='1-5', hour=9, minute=20)
        if not self.sched.running:
            self.sched.start()

    async def self_test(self):
        await emit.emit(Events.self_test)

    def _add_watch(self, plot: str, job_name: str, trigger: str, **func_args):
        """

        Args:
            plot: the name of the monitor
            job_name: the id of the job
            trigger: 如'interval:3' or 'frame:30m:-3'
            **func_args:

        Returns:

        """
        mon = create_plot(plot)

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

    def make_job_name(self, hash_keys, plot, trigger, **kwargs):
        data = kwargs.copy()
        data.update({
            "plot":    plot,
            "trigger": trigger
        })

        keys = []
        for key in hash_keys:
            keys.append(str(data.get(key)))

        return ":".join(keys)

    async def watch(self, plot: str, trigger: str, hash_keys: tuple = None, **kwargs):
        """
        (plot, code, frame_type, flag)构成惟一的一个监控任务。
        Args:
            plot:
            trigger: format like 'interval:3' or 'frame:30m:-3'
            hash_keys: the keys for generate an unique job name, if none, use (plot,
            code, frame_type, flag)
            **kwargs:

        Returns:

        """
        code = kwargs.get("code")
        frame_type = kwargs.get("frame_type")
        flag = kwargs.get("flag")

        if not all([code, frame_type, flag, plot]):
            raise ValueError("missing code, frame_type, flag or plot")

        hash_keys = hash_keys or ("plot", "code", "frame_type", "flag")
        job_name = self.make_job_name(hash_keys, plot, trigger, **kwargs)

        persist_data = {
            "jobinfo": {
                "plot":      plot,
                "trigger":   trigger,
                "hash_keys": hash_keys
            },
            "kwargs":  kwargs
        }

        # remove old ones first
        for job in self.sched.get_jobs():
            if job.name == job_name:
                self.sched.remove_job(job.id)
                await cache.sys.hdel(self.monitor_key, job_name)
                del self.watch_list[job.name]

        self.watch_list[job_name] = persist_data
        self._add_watch(plot, job_name, trigger, **kwargs)
        await cache.sys.hset(self.monitor_key, job_name, pickle.dumps(persist_data,
                                                                      protocol=0))

        return await self.list_watch(plot=plot, code=code)

    async def add_batch(self, plot: str, trigger: str, hash_keys: tuple = None,
                        **kwargs):
        """
        todo: 报告错误的参数，比如股票代码。
        todo: 回显已加的监控，含已存在的。
        Args:
            plot:
            trigger:
            **kwargs:

        Returns:

        """
        codes = kwargs.get("code_list")
        params = kwargs
        del params["code_list"]
        result = []
        for code in codes.split(","):
            params['code'] = code
            result.append(await self.watch(plot, trigger, hash_keys, **params))

    async def resume_monitors(self):
        """
        resume monitors from database, in case of the process is restarted
        Returns:

        """
        logger.info("loading monitor...")
        recs = await cache.sys.hgetall(self.monitor_key)
        for job_name, params in recs.items():
            params = pickle.loads(params.encode('utf-8'))
            plot = params.get("jobinfo").get("plot")
            trigger = params.get("jobinfo").get("trigger")

            kwargs = params.get("kwargs")
            self._add_watch(plot, job_name, trigger, **kwargs)
            self.watch_list[job_name] = params
        logger.info("done with %s monitor loaded", len(self.watch_list))

        return self.watch_list

    async def remove(self, job_name: str, plot: str = None, code: str = None,
                     frame_type: str = None,
                     flag: str = None,
                     remove_all=False):
        removed = []
        if remove_all:
            self.sched.remove_all_jobs()
            await cache.sys.delete(self.monitor_key)
            removed = self.watch_list
            self.watch_list = {}
            return removed
        else:
            if job_name:
                for job in self.sched.get_jobs():
                    if job.name == job_name:
                        removed.append(await self._remove(job.id, job.name))
                return removed
            else:
                pattern = rf"({plot})|({code})|({frame_type})|({flag})"
                for job in self.sched.get_jobs():
                    if re.search(pattern, job.name):
                        removed.append(await self._remove(job.id, job.name))

            return removed

    async def _remove(self, job_id, job_name):
        self.sched.remove_job(job_id)
        await cache.sys.hdel(self.monitor_key, job_name)
        del self.watch_list[job_name]
        return job_name

    async def list_watch(self, code: str = '', frame_type: str = '', plot: str = '',
                         flag: str = ''):
        pattern = rf"({plot})|({code})|({frame_type})|({flag})"

        results = []
        for job_name in self.watch_list.keys():
            if re.search(pattern, job_name):
                trigger = self.watch_list[job_name].get("jobinfo").get("trigger")
                plot = self.watch_list[job_name].get("jobinfo").get("plot")

                params = self.watch_list[job_name].get("kwargs")

                results.append([plot, params, trigger])

        return results


monitor = Monitor()

__all__ = ['monitor']
