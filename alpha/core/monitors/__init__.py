#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

from alpha.core.monitors.manager import MonitorManager

logger = logging.getLogger(__name__)


mm = MonitorManager()

__all__ = ['mm']