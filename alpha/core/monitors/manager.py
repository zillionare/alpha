#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import datetime
import json
import logging
import re

import cfg4py
from apscheduler.job import Job
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from omicron.core.triggers import TradeTimeIntervalTrigger, FrameTrigger
from omicron.core.types import FrameType
from omicron.dal import cache
from pyemit import emit

from alpha.core.enums import Events
from alpha.plots import create_plot

logger = logging.getLogger(__name__)

cfg = cfg4py.get_instance()


class MonitorManager:
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
        self.sched.add_job(self.resume_monitors, 'date', misfire_grace_time=30)
        trigger = FrameTrigger(FrameType.DAY, jitter=f"-6h")
        self.sched.add_job(self.self_test, trigger)
        if not self.sched.running:
            self.sched.start()

    async def self_test(self):
        await emit.emit(Events.self_test)

    def _add_watch(self, plot, job_name: str, job_info: dict):
        """

        Args:
            plot: instance of baseplot
            job_info: contains plot(the name), trigger(dict), title_keys, executor,
            params which is needed by executor

        Returns:

        """
        trigger = job_info.get('trigger')
        trigger_name = trigger.get("name")
        if trigger_name == "interval":
            interval = trigger.get('interval')
            unit = trigger.get('unit')
            _trigger = TradeTimeIntervalTrigger(f"{interval}{unit}")
        elif trigger_name == 'frame':
            frame_type = trigger.get("frame_type")
            jitter = trigger.get("jitter")
            jitter_unit = trigger.get("jitter_unit")
            _trigger = FrameTrigger(frame_type, f"{jitter}{jitter_unit}")
        else:
            raise ValueError(f"trigger type {trigger} not supported")

        executor = getattr(plot, job_info.get("executor"))
        self.sched.add_job(executor, trigger=_trigger, name=job_name,
                           kwargs=job_info.get("executor_params"),
                           misfire_grace_time=10)

    def find_job(self, plot, code, flag, frame_type: FrameType, *args):
        for job in self.sched.get_jobs():
            items = job.name.split(":")
            try:
                items.index(plot)
                items.index(code)
                items.index(flag)
                items.index(frame_type.value)
                for arg in args:
                    items.index(arg)

                return job

            except ValueError:
                continue

    def reschedule_job(self, start_time: datetime.datetime, job: Job):
        """
        I don't know why, but it doesn't work if call job's reschedule with different
        start_date. So I have to re-create the job
        Args:
            start_time:
            job_id:

        Returns:

        """
        job_info = self.watch_list.get(job.name)
        plot_name = job_info.get("plot")
        plot = create_plot(plot_name)

        self.sched.remove_job(job.id)
        self.sched.add_job(self._add_watch, 'date', next_run_time=start_time,
                           args=(plot, job.name, job_info),
                           misfire_grace_time=30)

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

    async def add_monitor(self, plot_name: str, **kwargs):
        """
        Args:
            plot_name: the name of plot
            kwargs: required by plot

        Returns:
        """
        plot = create_plot(plot_name)
        title_keys, job_info = plot.parse_monitor_settings(**kwargs)

        job_name = self.make_job_name(title_keys, plot_name, **kwargs)

        # remove old ones first
        for job in self.sched.get_jobs():
            if job.name == job_name:
                self.sched.remove_job(job.id)
                await cache.sys.hdel(self.monitor_key, job_name)
                del self.watch_list[job.name]

        self.watch_list[job_name] = job_info
        self._add_watch(plot, job_name, job_info)
        await cache.sys.hset(self.monitor_key, job_name, json.dumps(job_info))

    async def add_batch(self, plot_name: str, codes: str, **kwargs):
        """
        todo: 报告错误的参数，比如股票代码。
        todo: 回显已加的监控，含已存在的。
        Args:
            plot_name:
            trigger:
            **kwargs:

        Returns:

        """
        params = kwargs
        result = []
        for code in codes.split(","):
            result.append(await self.add_monitor(plot_name, code, **params))

    async def resume_monitors(self):
        """
        resume monitors from database, in case of the process is restarted
        Returns:

        """
        logger.info("(re)loading monitor...")

        jobs = await cache.sys.hgetall(self.monitor_key)
        for job_name, job_info in jobs.items():
            job_info = json.loads(job_info.encode('utf-8'))
            plot_name = job_info.get("plot")
            plot = create_plot(plot_name)
            self._add_watch(plot, job_name, job_info)
            self.watch_list[job_name] = job_info
        logger.info("done with %s monitor loaded", len(self.watch_list))

        return self.watch_list

    async def remove(self, job_name: str = None, plot: str = None, code: str = None,
                     frame_type: str = None,
                     flag: str = None,
                     remove_all=False):
        removed = []
        if remove_all:
            self.sched.remove_all_jobs()
            await cache.sys.delete(self.monitor_key)
            removed = self.watch_list.keys()
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

    async def list_monitors(self, code: str = '', frame_type: str = '', plot: str = '',
                            flag: str = ''):
        filters = filter(None, (code, frame_type, plot, flag))
        pattern = "|".join(f"({key})" for key in filters)

        results = []
        for job_name in self.watch_list.keys():
            if re.search(pattern, job_name):
                job_info = self.watch_list[job_name]
                plot = job_info.get("plot")
                trigger = job_info.get("trigger")
                params = job_info.get("executor_params")

                results.append([job_name, plot, params, trigger])

        return results

    def parse_trigger(self, trigger: str):
        if trigger.startswith('interval'):
            matched = re.match(r"interval:(\d+)([hmsd])", trigger)
            if matched:
                interval, unit = matched.groups()
                return "interval", interval, unit, None
            else:
                logger.warning("malformed triggers: %s", trigger)
                return None
        elif trigger.startswith('frame'):
            matched = re.match(r"frame:(\d+)([mdw]):?(.*)", trigger)
            if matched:
                interval, unit, jitter = matched.groups()
                return "frame", interval, unit, jitter
            else:
                logger.warning("malformed triggers: %s", trigger)
                return None

    def translate_trigger(self, trigger: dict):
        """
        将trigger转换成适于人阅读的格式
        Args:
            trigger: 包含name, interval, frame_type, unit, jitter, jitter_unit键

        Returns:

        """
        name = trigger.get("name")
        interval = trigger.get("interval")
        unit = trigger.get("unit")
        frame_type = trigger.get("frame_type")
        jitter = trigger.get("jitter")
        jitter_unit = trigger.get("jitter_unit")

        _time_unit_map = {
            "m": "分钟",
            "h": "小时",
            "d": "天",
            "s": "秒",
            "w": "周"
        }

        if name == 'interval':
            _translated = f"每{interval}{_time_unit_map[unit]}运行一次"
            return _translated

        if frame_type == '30m':
            _translated = "每30分钟运行一次"
        elif frame_type == '60m':
            _translated = "每60分钟运行一次"
        elif frame_type == "120m":
            _translated = "每120分钟运行一次"
        elif frame_type == "1d":
            _translated = "每交易日运行一次"
        elif frame_type == "1w":
            _translated = "每周交易结束时运行一次"
        elif frame_type == "1M":
            _translated = "每月交易结束时运行一次"
        else:
            raise ValueError(f"frame_type {frame_type} is not supported")

        if jitter is not None:
            if jitter < 0:
                _translated = f"{_translated}/每次提前{abs(jitter)}" \
                              f"{_time_unit_map[jitter_unit]}"
            elif jitter > 0:
                _translated = f"{_translated}/每次推迟{jitter}" \
                              f"{_time_unit_map[jitter_unit]}"

        return _translated
