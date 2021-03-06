#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

logger = logging.getLogger(__name__)


class CurveType:
    UNKNOWN = -1
    LINE_UP = 0
    LINE_DOWN = 1
    PARA_UP = 2
    PARA_DOWN = 3


class Events:
    sig_trade = "alpha/signals/trade"
    self_test = "alpha/signals/self_test"
    plot_pool = "alpha/plots/pool"
