#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging

import arrow
import jqdatasdk as jq

logger = logging.getLogger(__name__)

class Stocks:
    def __init__(self):
        jq.auth('18694978299', '8Bu8tcDpEAHJRn')
        self.secs = jq.get_all_securities()

    def all_stocks(self):
        return self.secs.index


    def get_bars(self, code, count, frame, end_dt=None):
        if end_dt is None:
            end_dt = arrow.now().datetime
        if isinstance(end_dt, arrow.Arrow):
            end_dt =  end_dt.datetime
        if not (code.upper().endswith('.XSHE') or code.upper().endswith('.XSHG')):
            if code.startswith('60'):
                code += '.XSHG'
            else:
                code += '.XSHE'
        fields = ['date', 'open', 'high', 'low', 'close', 'volume', 'money', 'factor']
        return jq.get_bars(code, count, frame, fields, include_now=True, end_dt=end_dt,
                           fq_ref_date=end_dt)


    def name_of(self, code: str):
        return self.secs[self.secs.index == code]['display_name'].iat[0]


    def check_buy_limit(self, c1: float, c: float, display_name: str = None):
        """
        股价是否达到涨停价
        :param display_name:
        :param c1: 前一日收盘价
        :param c: 当日收盘价
        :return:
        """
        if display_name and display_name.find("ST") != -1:
            limit = 0.05
        else:
            limit = 0.1

        if not all([c, c1]):
            return False

        return (c + 0.01) / c1 - 1 > limit

stocks = Stocks()

__all__ = ['stocks']