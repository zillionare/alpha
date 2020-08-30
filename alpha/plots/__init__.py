#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

from omicron.core.frametrigger import FrameTrigger
from omicron.core.types import FrameType

logger = logging.getLogger(__name__)


def create_plot(plot_name: str):
    if plot_name == 'momentum':
        from alpha.plots.momentum import Momentum
        return Momentum()
    if plot_name == 'maline':
        from alpha.plots.maline import MALinePlot
        return MALinePlot()
    if plot_name == 'fix_price':
        from alpha.plots.fixprice import FixPrice
        return FixPrice()
    if plot_name == 'extendline':
        from alpha.plots.extendline import ExtendLine
        return ExtendLine()


__all__ = ['create_plot']
