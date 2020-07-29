#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import asyncio
import atexit
import datetime
import logging
import os
import random
from typing import List, Optional

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
from sklearn import svm, metrics
from joblib import dump, load
from sklearn.model_selection import GridSearchCV

from alpha.core.enums import CurveType
from alpha.core.signal import moving_average, polyfit, rmse
from alpha.config import get_config_dir
from alpha.config.cfg4py_auto_gen import Config

logger = logging.getLogger(__name__)
cfg: Config = cfg4py.get_instance()


class MomentumPlot:
    def __init__(self):
        self.model = None

    async def init(self, model_file:Optional[str]=None):
        os.environ[cfg4py.envar] = 'DEV'
        self.config_dir = get_config_dir()

        cfg4py.init(self.config_dir, False)
        await emit.start(engine=emit.Engine.REDIS, dsn=cfg.redis.dsn,
                          start_server=True)
        await omicron.init()

        if model_file is not None:
            with open(model_file, "rb") as f:
                self.model = load(f)

    async def exit(self):
        await emit.stop()

    def extract_features(self, bars, max_error):
        """

        Args:
            bars:

        Returns:

        """
        ma5 = moving_average(bars['close'], 5)[-5:]
        ma10 = moving_average(bars['close'], 10)[-5:]
        ma20 = moving_average(bars['close'], 20)[-5:]

        feat = []
        for ts in [ma5, ma10, ma20]:
            try:
                err, curve, coef = polyfit(ts, CurveType.PARABOLA)
                # 取系数a和b。对于一次函数，a=0;对于二次函数和指数函数，a,b都存在。c为截距，忽略
                a, b = coef[0], coef[1]
                if np.isnan(a) or np.isnan(b) or np.isnan(err) or err > max_error:
                    raise ValueError
                feat.extend([a, b, -b/(2*a)])
            except Exception:
                return feat

        return feat

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

    async def _build_train_data(self, n: int, max_error:float=0.01):
        """
        从最近的符合条件的日期开始，遍历股票，提取特征和标签，生成数据集。
        Args:
            n: 需要采样的样本数

        Returns:

        """
        watch_win = 1
        max_curve_len = 5
        max_ma_win = 20

        #y_stop = arrow.get('2020-7-24').date()
        y_stop = arrow.now().date()
        y_start = tf.day_shift(y_stop, -watch_win + 1)
        x_stop = tf.day_shift(y_start, -1)
        x_start = tf.day_shift(x_stop, -(max_curve_len + max_ma_win - 1))
        data = []
        while len(data) < n:
            #for code in Securities().choose(['stock']):
            for code in ['000601.XSHE']:
                try:
                    sec = Security(code)
                    x_bars = await sec.load_bars(x_start, x_stop, FrameType.DAY)
                    y_bars = await sec.load_bars(y_start, y_stop, FrameType.DAY)
                    # [a, b, axis] * 3
                    x = self.extract_features(x_bars, max_error)
                    if len(x) == 0: continue
                    y = np.max(y_bars['close'])/x_bars[-1]['close'] - 1
                    if np.isnan(y): continue

                    feature = [code, tf.date2int(x_stop)]
                    feature.extend(x)
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
    async def build_train_data(self, save_to:str, n=100)->List:
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

    @async_run
    async def screen(self, model_file:str):
        """筛选"""
        await self.init(model_file)
        for date in ['2020-7-21', '2020-7-22', '2020-7-23', '2020-7-24']:
            date = arrow.get(date).date()
            for code in ['000982.XSHE']:
                result = await self.predict(code, date)
                print(result)

    @async_run
    async def train(self, save_to:str, dataset:str=None, n:int=100):
        """

        Args:
            dataset:

        Returns:

        """
        await self.init()
        save_to = os.path.abspath(save_to)
        if not os.path.exists(save_to):
            logger.warning("invalid path: %s", save_to)
            return

        date = tf.date2int(arrow.now().date())
        save_to = f"{save_to}/momemtum.{date}.svm"

        x_train, y_train = [], []
        x_test, y_test = [], []
        if os.path.exists(dataset):
            with open(dataset, 'r') as f:
                data = f.readlines()[1:n+1]
                random.shuffle(data)
                n_train = int(len(data) * 0.8)
                for line in data[:n_train]:
                    fields = line.strip("\n").split("\t")
                    x_train.append(list(map(lambda x: float(x), fields[2:-1])))
                    y_train.append(float(fields[-1]))

                for line in data[n_train:]:
                    fields = line.strip("\n").split("\t")
                    x_test.append(list(map(lambda x: float(x), fields[2:-1])))
                    y_test.append(float(fields[-1]))

        else:
            data = await self._build_train_data(n)
            random.shuffle(data)
            n_train = int(len(data) * 0.8)
            for rec in data[:n_train]:
                x_train.append(rec[2:-1])
                y_train.append(rec[-1])
            for rec in data[n_train:]:
                x_test.append(rec[2:-1])
                y_test.append(rec[-1])

        assert len(x_train) == len(y_train)
        logger.info("train data loaded, %s records in total", len(x_train))
        params = {
            'C': [1e-3, 1e-2, 1e-1, 1, 10],
            'kernel': ('linear', 'poly'),
            'gamma': [0.001,0.005,0.1,0.15,0.20,0.23,0.27],
            'epsilon': [1e-4, 1e-3, 1e-2,1e-1,1,10],
            'degree': [2,3]
        }
        clf = GridSearchCV(svm.SVR(verbose=True), params, n_jobs=-1)
        clf.fit(x_train, y_train)
        logger.info("Best: %s, %s, %s", clf.best_estimator_, clf.best_score_,
                    clf.best_params_)
        self.model = clf.best_estimator_
        y_pred = self.model.predict(x_test)
        score = rmse(y_test, y_pred)
        logger.info("training score is:%s", score)
        print("y_true, y_pred")
        for yt, yp in zip(y_test[:20], y_pred[:20]):
            print(yt, yp)
        with open(save_to, "wb") as f:
            dump(self.model, f)

        await self.exit()

    async def predict(self, code, x_end_date: datetime.date, max_error:float=0.01):
        sec = Security(code)
        start = tf.day_shift(x_end_date, -29)
        bars = await sec.load_bars(start, x_end_date, FrameType.DAY)
        features = self.extract_features(bars, max_error)
        if len(features) == 0:
            logger.warning("cannot extract features from %s(%s)", code, x_end_date)
        else:
            return self.model.predict([features])



if __name__ == "__main__":
    m = MomentumPlot()
    fire.Fire({
        "build_train_data": m.build_train_data,
        "train": m.train,
        "screen": m.screen
    })
