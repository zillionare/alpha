#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import datetime
import logging
import os
import pickle
from typing import List

import arrow
import cfg4py
import fire
import numpy as np
import omicron
from omicron.core.lang import async_run
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security
from pyemit import emit
from sklearn import svm

from alpha.core.enums import CurveType
from alpha.core.signal import moving_average, polyfit
from config import get_config_dir
from config.cfg4py_auto_gen import Config

logger = logging.getLogger(__name__)
cfg: Config = cfg4py.get_instance()


class MomentumPlot:
    def __init__(self):
        self.model = None
    async def init(self):
        os.environ[cfg4py.envar] = 'DEV'
        self.config_dir = get_config_dir()

        cfg4py.init(self.config_dir, False)
        await emit.start(engine=emit.Engine.REDIS, dsn=cfg.redis.dsn, start_server=True)
        await omicron.init()

    def extract_features(self, bars):
        """

        Args:
            bars:

        Returns:

        """
        ma5 = moving_average(bars['close'], 5)[-5:]
        ma10 = moving_average(bars['close'], 10)[-5:]
        ma20 = moving_average(bars['close'], 20)[-5:]

        fit = []
        for ts in [ma5, ma10, ma20]:
            try:
                err, curve, coef = polyfit(ts, CurveType.PARABOLA)
                # 取系数a和b。对于一次函数，a=0;对于二次函数和指数函数，a,b都存在。c为截距，忽略
                fit.extend([coef[0], coef[1], err])
            except Exception as e:
                pass

        return fit

    async def find_by_moving_average(self):
        result = []
        for code in Securities().choose(['stock']):
            day = arrow.now().date()
            sec = Security(code)
            try:
                signal, fit = await self.test_signal(sec, day)
                if abs(signal) == 1:
                    result.append([code, day, signal, fit])
            except Exception as e:
                logger.info(e)
                continue
            # reporter.info("%s,%s,%s,%s,%s,%s,%s,%s,%s",
            #               code, day, signal, *fit[0], *fit[1], *fit[2])

        return result

    async def _build_train_data(self, n: int):
        """
        从最近的符合条件的日期开始，遍历股票，提取特征和标签，生成数据集。
        Args:
            n: 需要采样的样本数

        Returns:

        """
        watch_win = 2
        max_curve_len = 10
        max_ma_win = 20

        #y_stop = arrow.get('2020-7-24').date()
        y_stop = arrow.now().date()
        y_start = tf.day_shift(y_stop, -watch_win + 1)
        x_stop = tf.day_shift(y_start, -1)
        x_start = tf.day_shift(x_stop, -(max_curve_len + max_ma_win - 1))
        data = []
        while len(data) < n:
            for code in Securities().choose(['stock']):
            #for code in ['000982.XSHE']:
                try:
                    sec = Security(code)
                    x_bars = await sec.load_bars(x_start, x_stop, FrameType.DAY)
                    y_bars = await sec.load_bars(y_start, y_stop, FrameType.DAY)
                    x = self.extract_features(x_bars)
                    if len(x) == 0 or np.isnan(x[0]):
                        continue
                    y = np.max(y_bars['close'])/x_bars[-1]['close'] - 1
                    if np.isnan(y): continue
                    x.append(y)

                    feature = [code, tf.date2int(x_stop)]
                    feature.extend(x)
                    # [a, b, err] * 3 + y
                    data.append(feature)
                except Exception as e:
                    logger.warning("Failed to extract features for %s (%s)",
                                   code,
                                   x_stop)
                    logger.exception(e)
                if len(data) >= n:
                    break
                if len(data) % 500 == 0:
                    logger.info("got %s records.", len(data))
            y_stop = tf.day_shift(y_stop, -1)
            y_start = tf.day_shift(y_start, -1)
            x_stop = tf.day_shift(y_start, -1)
            x_start = tf.day_shift(x_start, -1)

        return data


    @async_run
    async def build_train_data(self, save_to:str, n=10)->List:
        await self.init()
        data = await self._build_train_data(n)
        date = tf.date2int(arrow.now().date())
        path = os.path.abspath(save_to)
        path = os.path.join(path, f"momemtum.{date}.tsv")
        with open(path, "w") as f:
            cols = "code,date,a5,b5,err5,a10,b10,err10,a20,b20,err20,y".split(",")
            f.writelines("\t".join(cols))
            f.writelines("\n")
            for item in data:
                f.writelines("\t".join(map(lambda x:str(x), item)))
                f.writelines("\n")

        return data


    async def _screen(self):
        pass

    def screen(self):
        """筛选"""
        pass

    @async_run
    async def train(self, save_to:str, dataset:str=None, n:int=100):
        """

        Args:
            dataset:

        Returns:

        """
        save_to = os.path.abspath(save_to)
        if not os.path.exists(save_to):
            logger.warning("invalid path: %s", save_to)
            return
        date = arrow.now().date()
        save_to = f"{save_to}/momemtum.{date}.svm"

        x_train, y_train = [], []
        if os.path.exists(dataset):
            with open(dataset, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    if i == 0: continue # skip header
                    fields = line.strip("\n").split("\t")
                    x_train.append(fields[2:-1])
                    y_train.append(fields[-1])
        else:
            data = await self._build_train_data(n)
            for rec in data:
                x_train.append(rec[2:-1])
                y_train.append(rec[-1])

        assert len(x_train) == len(y_train)
        logger.info("train data loaded, %s records in total", len(x_train))
        self.model = svm.SVC()
        self.model.fit(x_train[:-200], y_train[:-200])
        logger.info("model trained")
        score = self.model.score(x_train[-200:], y_train[-200:])
        logger.info("training score is:%s", score)
        with open(save_to, "wb") as f:
            pickle.dump(self.model, f, protocol=4)


    async def predict(self, code, x_end_date: datetime.date):
        sec = Security(code)
        start = tf.day_shift(x_end_date, -29)
        bars = await sec.load_bars(start, x_end_date, FrameType.DAY)
        features = self.extract_features(bars)
        if len(features) and not np.isnan(features[0]):
            self.model.predict(features)
        else:
            logger.warning("failed to extract feature from %s(%s)", code, x_end_date)



if __name__ == "__main__":
    m = MomentumPlot()
    fire.Fire({
        "build_train_data": m.build_train_data,
        "train": m.train
    })
