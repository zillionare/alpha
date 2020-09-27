#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

from omicron.core.triggers import FrameTrigger
from omicron.core.types import FrameType

logger = logging.getLogger(__name__)


def create_plot(plot_name: str):
    if plot_name.lower() == 'momentum':
        from alpha.plots.momentum import Momentum
        return Momentum()
    if plot_name.lower() == 'maline':
        from alpha.plots.maline import MaLine
        return MaLine()
    if plot_name.lower() == 'fixprice':
        from alpha.plots.fixprice import FixPrice
        return FixPrice()
    if plot_name.lower() == 'extendline':
        from alpha.plots.extendline import ExtendLine
        return ExtendLine()


def start_plot_scan(scheduler):
    mom = create_plot('momentum')
    # 每个交易日14：30，选出日线级别符合动量策略的股票
    trigger = FrameTrigger(FrameType.DAY, "-30m")
    scheduler.add_job(mom.scan, trigger, frame_type=FrameType.DAY)

    trigger = FrameTrigger(FrameType.MIN30)
    scheduler.add_job(mom.scan, trigger, frame_type=FrameType.MIN30)


__all__ = ['create_plot']
