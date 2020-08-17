#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import enum
import logging

logger = logging.getLogger(__name__)

class CurveType(enum.IntEnum):
    LINE = 0
    PARABOLA = 1
    EXP = 1

class Events:
    sig_long = "alpha/signals/long"
    sig_short = "alpha/signals/short"