import asyncio
import functools
import logging
import os
import pickle
from math import copysign, e

import arrow
import cfg4py
import fire
import numpy as np
import omicron
import pandas as pd
from numpy.typing import ArrayLike
from omicron import cache
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.models.security import Security
from scipy.stats import randint, uniform
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBClassifier

from alpha.core.features import (
    fillna,
    moving_average,
    relation_with_prev_high,
    relationship_with_prev_low,
    relative_strength_index,
    top_n_argpos,
    volume_features,
    weighted_moving_average,
)
from alpha.core.morph import MorphFeatures
from alpha.core.smvecstore import SmallSizeVectorStore
from alpha.utils import round
from alpha.utils.data.databunch import DataBunch

cfg = cfg4py.init("/apps/alpha/alpha/config")
logger = logging.getLogger(__name__)


class Twins:
    """寻找相似图形来选股的策略。"""

    def __init__(self, name: str, *args, **kwargs):
        self.name = name

        self.nbars = kwargs.get("nbars", 81)
        self.metric = "L2"
        self.ma_wins = [5, 10, 20, 60]
        self.vol_win = 80
        self.flen = 20

        self.day_morph = MorphFeatures.load(FrameType.DAY)

        self.X = []
        self.y = []
        self.meta = []

        # the machine learning model
        self.model = None
        self.version = 1

        os.makedirs(os.path.expanduser(f"~/alpha/data/twins"), exist_ok=True)

    def __str__(self):
        bins = [-10, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 10]
        count, bins = np.histogram(self.y, bins=bins)

        count = "".join(map(lambda x: f"{x:>7}", count))
        bins = "".join(map(lambda x: f"{x:>7}", bins))

        dist = f"{bins}\n  {count}"
        return (
            f"Twins {self.name}\n patterns: {len(self.X)}, with distrubutions:\n{dist}"
        )

    @classmethod
    def from_model(model: str) -> "Twins":
        if not os.path.exists(model):
            model = os.path.expanduser(f"~/alpha/data/twins/{model}.pkl")

        with open(model, "rb") as f:
            return pickle.load(f)

    def find_pattern(self, code: str, end: str, frame_type: str = "1d"):
        meta = np.array(
            self.meta, dtype=[("code", "S11"), ("end", "U20"), ("frame_type", "S4")]
        )

        found = meta
        for col, val in (("code", code), ("frame_type", frame_type), ("end", end)):
            found = found[found[col] == val]

        return found

    async def add_pattern(
        self, op: int, code: str, end: str, frame_type: str = "1d"
    ) -> None:
        if self.find_pattern(code, end, frame_type).shape[0] > 0:
            return "already exists"

        sec = Security(self.canonicalize(code))
        end = arrow.get(end)
        ft = FrameType(frame_type)

        start = tf.shift(end, -self.nbars + 1, ft)
        bars = await sec.load_bars(start, end, ft)
        if (
            np.count_nonzero(np.isfinite(bars["close"])) < len(bars) * 0.9
            or len(bars) < self.nbars
        ):
            return None

        vec = self.x_transform(bars)

        self.meta.append((code, end, frame_type))
        self.X.append(vec)
        self.y.append(op)

        return self._describe_vec(vec)

    def _describe_vec(self, vec):
        """对一个特征向量进行描述

        Args:
            vec ([type]): [description]

        Returns:
            [type]: [description]
        """
        # 成交量特征: 方向 3, 间隔 3， 量比 3
        vol = f"成交量： 方向({self.vol_win}): {vec[:3]} 间隔: {np.round(vec[3:6], 2)} 量比: {np.round(vec[6:9], 2)}"

        # RSI特征: 最后4周期
        rsi = f"RSI<最后4>: {np.round(vec[9:13], 2)}"

        # 与前高关系
        prev_high = f"前高({max(self.ma_wins)}): {vec[13]:.1f}<tan> 前高比: {vec[14]:.0%}"

        # 与前低关系
        prev_low = f"前低({max(self.ma_wins)}): {vec[15]:.1f}<tan> 前低比: {vec[16]:.0%}"

        # 发散程度
        div = f"发散程度: {vec[17]:.2f}"

        return f"{vol}\n{rsi}\n{prev_high}\n{prev_low}\n{div}"

    async def add_patterns(self, path: str):
        """add pattern from file

        Args:
            path (str): path to file
        """
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                op, code, end, desc = line.strip().split("\t")
                await self.add_pattern(int(op), code, end, desc)

    def canonicalize(self, code: str):
        if code.endswith(".XSHE") or code.endswith(".XSHG"):
            return code

        if code.startswith("6"):
            return code + ".XSHG"
        else:
            return code + ".XSHE"

    def x_transform(self, bars):
        """
        1. 均线使用5, 10, 20, 60各`flen`根,计算ma,及ma之间的发散程度。共需要66个周期的close
        2. 计算60周期以来最大的3次成交量的方向、间隔，如果成交量大于成交均量1倍
        3. 计算RSI中低于20，大于80时，距当前的距离tanh(i*2/len(bars))
        4. 成交价特征：最后10周期阴阳线特征和涨跌幅特征
        Args:
            bars ([type]): [description]
        """
        vec = []
        close = bars["close"]

        close = fillna(close.copy())

        # 成交量特征
        vec.extend(volume_features(bars, self.vol_win))

        # RSI
        rsi = relative_strength_index(close)[-4:]
        vec.extend(rsi)

        # 与前高的关系
        n = max(self.ma_wins)
        vec.extend(relation_with_prev_high(close[-n:], n))

        # 与前低的关系
        vec.extend(relationship_with_prev_low(close[-n:], n))

        mas = []
        for win in self.ma_wins:
            ma = moving_average(close, win)
            mas.append(ma[-1])

        # 各均线的发散程度？太发散则有回归需求；太聚拢则有发散需求
        divergency = (mas[0] - mas[-1]) / (mas[0] + mas[-1])
        vec.append(divergency)

        # len(vec) == 18 till now
        # moving average line's morph feature, dim is 4
        vec.extend(self.day_morph.encode(close))

        # 成交价变化特征
        vec.extend(self.price_features(close, self.flen))

        return vec

    def price_features(self, close: np.array, flen: int = 10):
        """计算收盘价的特征， 最近`flen`根收盘价的变化率，即最高价位置

        Args:
            close (np.array): 收盘价序列
            flen (int, optional): [description]. Defaults to 10.
        """
        vec = []
        nbars = len(close)

        vec.extend((close[1:] / close[:-1] - 1)[-flen:])

        top_prices = top_n_argpos(close, 3)
        vec.extend(np.tanh(2 * (nbars - 1 - top_prices) / nbars))

        return vec

    async def train(self, datafile: str, params=None):
        """训练模型"""
        params = params or {
            "colsample_bytree": uniform(0.7, 0.3),
            "gamma": uniform(0, 0.5),
            "learning_rate": uniform(0.01, 1),
            "max_depth": randint(2, 6),
            "n_estimators": randint(80, 150),
            "subsample": uniform(0.6, 0.4),
        }

        model = XGBClassifier()

        search = RandomizedSearchCV(
            model,
            param_distributions=params,
            random_state=78,
            n_iter=200,
            cv=10,
            verbose=1,
            n_jobs=1,
            return_train_score=True,
            refit=True,  # do the refit oursel
        )

        ds = DataBunch(self.X, self.y)
        ds.train_test_split()
        fit_params = {"eval_set": [(ds.X_test, ds.y_test)], "early_stopping_rounds": 10}

        search.fit(ds.X_train, ds.y_train, **fit_params)
        model = search.best_estimator_
        preds = model.predict(ds.X_test)
        report = classification_report(ds.y_test, preds)

        return model, report

    async def predict(
        self,
        code: str,
        end: str,
        frame_type: str = "30m",
        threshold=None,
        top_n: int = 1,
    ):
        """预测"""
        if not self.stores:
            raise ValueError("please load store before make prediction")

        sec = Security(self.canonicalize(code))
        ft = FrameType(frame_type)

        if end is None:
            end = tf.day_shift(arrow.now(), 0)
            if ft in tf.minute_level_frames:
                end = tf.combine_time(end, hour=15)
        else:
            end = arrow.get(end)

        start = tf.shift(end, -self.nbars + 1, ft)

        bars = await sec.load_bars(start, end, ft)

        return self._predict(bars, threshold, top_n)

    def _predict(self, bars, threshold=1e-2, n: int = 1):
        """预测"""
        vec = self.normalize(self.x_transform(bars))

        if vec is None:
            return None

        res = self.store.nearest_vec(vec, threshold, metric=self.metric)
        if threshold is not None:
            return res[res["d"] < threshold][:n]
        else:
            return res[:n]

    def normalize(self, x):
        return x

    async def test(
        self,
        code: str,
        n: int = 10,
        end: str = None,
        frame_type: str = "30m",
        threshold=9999,
    ):
        """对`code`在`end`指定的前`n`个周期里，进行pattern匹配测试。

        `end`之后，需要留至少5个周期，以便观察。
        Args:
            code (str): 股票代码
            start (str): 开始时间
            n (int): 周期数
            frame_type (str): 周期类型
        """
        wwin = 10  # watch window

        code = self.canonicalize(code)
        sec = Security(code)
        frame_type = FrameType(frame_type)
        if end is None:
            end = tf.day_shift(arrow.now(), 0)
            if frame_type in tf.minute_level_frames:
                end = tf.combine_time(end, hour=15)

            end = tf.shift(end, -wwin + 1, frame_type)
        else:
            end = arrow.get(end)

        end = tf.shift(end, wwin - 1, frame_type)
        assert end < arrow.now(), f"end time must be later than now: {end}"

        start = tf.shift(end, -self.nbars - n - 2, frame_type)

        head, tail = await cache.get_bars_range(code, frame_type)
        if any([head is None, tail is None]):
            return None
        start = max(start, head)
        end = min(end, tail)

        if tf.count_frames(start, end, frame_type) < self.nbars + wwin:
            return None

        bars = await sec.load_bars(start, end, frame_type)

        results = []
        tstart = bars[self.nbars - 1]["frame"]
        tend = bars[-wwin - 1]["frame"]
        print(f"test {code} from {tstart} to {tend}")
        for i in range(n):
            if i + self.nbars + wwin > len(bars):
                break

            xbars = bars[i : i + self.nbars]
            ybars = bars[i + self.nbars : i + self.nbars + wwin]

            close = xbars["close"]

            if np.any(ybars["close"] == None) or np.count_nonzero(
                close == None
            ) > 0.1 * len(close):
                continue

            res = self._predict(xbars, threshold=threshold)
            # frame, operation, dist, profit, risk,  sample_code, sample_point, desc
            row = [xbars[-1]["frame"], code]
            if res is None or len(res) == 0:
                continue

            # operation
            row.append(res[0]["op"])
            flag = copysign(1, res[0]["op"])

            # distance
            row.append(res[0]["d"])

            # profit or gain of avoiding loss
            xclose = fillna(close)
            row.append(flag * (max(ybars["close"]) / xclose[-1] - 1))
            # risk if we act or not act
            row.append(min(ybars["low"]) / xbars["close"][-1] - 1)

            # sample_code, sample_point and desc
            row.extend([res[0]["code"], res[0]["end"], res[0]["desc"]])

            results.append(row)

        if len(results) == 0:
            return None

        return results

    def draw_test_report(
        self,
        results: list,
    ):
        opcode = {2: "买入", 1: "轻仓参与", 0: "持仓不动", -1: "减仓", -1: "清仓"}

        if len(results) == 0:
            print("no results")
            return None

        df = pd.DataFrame(
            results, columns=["时间", "代码", "操作", "误差", "收益", "风险", "模板", "取样点", "特征"]
        )

        return (
            df.style.set_properties(**{"text-align": "left"}, subset=("特征"))
            .set_table_styles([dict(selector="th", props=[("text-align", "center")])])
            .format(
                formatter={
                    "操作": lambda v: opcode.get(v),
                    "收益": "{:.2%}",
                    "风险": "{:.2%}",
                    "误差": "{:.2f}",
                    "时间": lambda t: f"{t.year}{t.month:02d}{t.day:02d} {t.hour:02d}:{t.minute:02d}",
                    "取样点": lambda t: f"{t.year}{t.month:02d}{t.day:02d} {t.hour:02d}:{t.minute:02d}",
                }
            )
            .hide_index()
            .apply(
                lambda x: ["background-color: #22aa22" if v < 3e-3 else "" for v in x],
                axis=1,
                subset=("误差"),
            )
            .apply(
                self.format_ops_cell,
                axis=1,
                subset=("操作"),
            )
        )

    def format_ops_cell(self, x):

        style = []
        for v in x:
            if v > 0:
                style.append("background-color: #cc2222")
            elif v == 0:
                style.append("background-color: #ffffff")
            else:
                style.append("background-color: #22aa22")

        return style

    def remove_pattern(self, code: str, end: str):
        """
        删除模式
        """
        return self.morph_store.remove("code", code, end=arrow.get(end))

    def save(self, model: str = None):
        if model is None:
            model = os.path.expanduser(f"~/alpha/data/twins/twins-v{self.version}.pkl")

        with open(model, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"twins has been saved to {model}")

    @staticmethod
    def load(model: str):
        if not os.path.exists(os.path.expanduser(model)) and model.startswith("v"):
            path = os.path.expanduser(f"~/alpha/data/twins/twins-{model}.pkl")
        else:
            path = os.path.expanduser(model)

        with open(path, "rb") as f:
            return pickle.load(f)


def async_run_command(func):
    async def _init_and_run(*args, **kwargs):
        cfg4py.init("/apps/alpha/alpha/config")
        await omicron.init()
        await func(*args, **kwargs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        asyncio.run(_init_and_run(*args, **kwargs))

    return wrapper


@async_run_command
async def train(datafile: str, version=None):
    s = Twins("sv-30m")
    await s.train(datafile, version)


@async_run_command
async def predict(
    model: str, code: str, end: str = None, ft: str = "30m", threshold=3e-2
):
    """[summary]

    Args:
        model (str): [description]
        code (str): [description]
        end (str, optional): [description]. Defaults to None.
        ft (str, optional): [description]. Defaults to '30m'.
        threshold ([type], optional): Defaults to 3e-2. according to test, even x == y, L2 (sqrt(dot(x,x) - 2 * dot(x,y) + dot(y,y)) will still get 0.0015. so it's better to choose 3e-2 ad threshold
    """
    s = Twins("sv-30m")

    if os.path.exists(model):
        s.load(model)
    elif model.lower().startswith("v"):
        path = os.path.expanduser(f"~/alpha/data/{s.name}/{s.name}-{model}.pkl")
        s.load(path)
    print(await s.predict(code, end, ft, threshold=threshold))


@async_run_command
async def test(
    model: str, code: str, n: int = 10, end: str = None, ft: str = "30m", threshold=3e-3
):
    s = Twins("sv-30m")

    if os.path.exists(model):
        s.load(model)
    elif model.lower().startswith("v"):
        path = os.path.expanduser(f"~/alpha/data/{s.name}/{s.name}-{model}.pkl")
        s.load(path)

    df = await s.test(code, n, end, ft, threshold)
    print(df)


if __name__ == "__main__":
    fire.Fire({"train": train, "predict": predict, "test": test})
