from enum import IntEnum
from alpha.core.smvecstore import SmallSizeVectorStore
import asyncio
import functools
import logging
import os
import sqlite3
from math import copysign
from typing import Callable, List

import arrow
import cfg4py
import fire
import numpy as np
import omicron
import pandas as pd
from numpy.typing import ArrayLike
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.models.security import Security
from pymilvus import DataType, Milvus
from sklearn.preprocessing import normalize
import plotly.graph_objects as go

from alpha.core.features import (
    fillna,
    moving_average,
    relative_strength_index,
    top_n_argpos,
)
from alpha.utils import round

cfg = cfg4py.init("/apps/alpha/alpha/config")
logger = logging.getLogger(__name__)


class SimVecStrategy:
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.store = SmallSizeVectorStore(
            name, {"op": "<i4", "code":"O", "end": "O", "desc": "O",}
        )
        self.nbars = kwargs.get("nbars", 81)
        self.metric = "L2"

    def load(self, path: str) -> None:
        self.store = SmallSizeVectorStore.load(path)

    async def add_pattern(
        self, op: int, code: str, end: str, desc: str, frame_type: str = "30m"
    ) -> None:
        sec = Security(self.canonicalize(code))
        end = arrow.get(end)
        ft = FrameType(frame_type)

        start = tf.shift(end, -self.nbars, ft)
        bars = await sec.load_bars(start, end, ft)
        if (
            np.count_nonzero(bars["close"] == None) > len(bars) * 0.9
            or len(bars) < self.nbars
        ):
            return None

        vec = self.xtransform(bars)
        self.store.insert({"op": op, "desc": desc, "end": end, "code": code}, vec)

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

    def ma_features(self, close: np.array, wins=(5, 10, 20, 60), flen: int = 7):
        # 1. 先进行归一化，以便我们纯粹只比较形态
        c_ = close / close[-1]

        vec = []
        mas = []
        for win in wins:
            ma = moving_average(c_, win)[-flen:]
            vec.extend(ma)
            mas.append(ma)

        # 各均线的发散程度？太发散则有回归需求；太聚拢则有发散需求
        mas = np.stack(mas)
        mas = np.max(mas, axis=0) - np.min(mas, axis=0)
        vec.extend(mas)

        return vec

    def volume_features(self, volume: np.array, flags: ArrayLike, win: int = 80):
        """计算`win`周期以来，最大3次成交量的方向和间隔，如果成交量大于成交均量的1倍

        Args:
            volume (np.array): [description]
            flags (ArrayLike): 当前成交量的方向
            win (int, optional): [description]. Defaults to 80.

        Raises:
            ValueError: [description]
        """
        if len(volume) < win or len(flags) < win:
            raise ValueError(f"len of volume/flags should be >= {win}")

        vec = []

        volume = volume[-win:]
        flags = flags[-win:]

        avg = np.mean(volume)
        indice = top_n_argpos(volume, 3)
        valid_indice = []
        for i in indice:
            if volume[i] > avg:
                valid_indice.append(i)

        indice = sorted(valid_indice)
        np.pad(indice, (0, 3 - len(indice)), "constant", constant_values=0)
        vec.extend(flags[indice])

        # 间隔
        vec.extend([np.tanh(2 * (win - i - 1) / win) for i in indice])
        return vec

    def rsi_features(self, close: np.array, rsi_win: int = 6):
        rsi = relative_strength_index(close, rsi_win)

        vec = []
        nbars = len(close)

        min_rsi_pos = np.argwhere(rsi < 20)
        if len(min_rsi_pos) > 0:
            min_rsi_pos = min_rsi_pos[-1] + rsi_win
            vec.extend(np.tanh((nbars - 1 - min_rsi_pos) * 2 / nbars))
        else:
            vec.append(1)

        max_rsi_pos = np.argwhere(rsi > 90)
        if len(max_rsi_pos) > 0:
            max_rsi_pos = max_rsi_pos[-1] + rsi_win
            vec.extend(np.tanh((nbars - 1 - max_rsi_pos) * 2 / nbars))
        else:
            vec.append(1)

        return vec

    def canonicalize(self, code: str):
        if code.endswith(".XSHE") or code.endswith(".XSHG"):
            return code

        if code.startswith("6"):
            return code + ".XSHG"
        else:
            return code + ".XSHE"

    def xtransform(self, bars, flen: int = 7):
        """
        1. 均线使用5, 10, 20, 60各7根,计算ma,及ma之间的发散程度。共需要66个周期的close
        2. 计算60周期以来最大的3次成交量的方向、间隔，如果成交量大于成交均量1倍
        3. 计算RSI中低于20，大于80时，距当前的距离tanh(i*2/len(bars))
        4. 成交价特征：最后10周期阴阳线特征和涨跌幅特征
        Args:
            bars ([type]): [description]
        """
        vec = []
        close = bars["close"]
        if np.count_nonzero(np.isfinite(close)) < len(close) * 0.9:
            return None

        close = fillna(close.copy())
        vec.extend(self.ma_features(close))

        volume = fillna(bars["volume"].copy())
        vec.extend(
            self.volume_features(volume, np.where(close[1:] > close[:-1], 1, -1))
        )

        # RSI
        vec.extend(self.rsi_features(close))

        # 成交价变化特征
        vec.extend((close[1:] / close[:-1])[-flen:])

        return vec

    async def train(self, datafile: str, version=None):
        """训练模型"""
        await self.add_patterns(datafile)

        save_to = os.path.expanduser(f"~/alpha/data/{self.name}/")
        os.makedirs(save_to, exist_ok=True)

        version = version or str(arrow.now().datetime)
        self.store.save(os.path.join(save_to, f"{self.name}-{version}.pkl"))

    async def predict(
        self,
        code: str,
        end: str,
        frame_type: str = "30m",
        threshold=None,
        top_n: int = 1,
    ):
        """预测"""
        if not self.store:
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

    def _predict(self, bars, threshold=None, n: int = 1):
        """预测"""
        vec = self.normalize(self.xtransform(bars))

        res = self.store.search_vec(vec, top_n=n, metric=self.metric)
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
        """对`code`在`start`指定的`n`个周期里，进行pattern匹配测试。
        Args:
            code (str): 股票代码
            start (str): 开始时间
            n (int): 周期数
            frame_type (str): 周期类型
        """
        sec = Security(self.canonicalize(str(code)))
        frame_type = FrameType(frame_type)
        if end is None:
            end = tf.day_shift(arrow.now(), 0)
            if frame_type in tf.minute_level_frames:
                end = tf.combine_time(end, hour=15)
        else:
            end = arrow.get(end)

        results = []
        for i in range(n // 4):
            tail = tf.shift(end, -i * 4, frame_type)
            head = tf.shift(tail, -80, frame_type)
            xbars = await sec.load_bars(head, tail, frame_type)
            close = xbars["close"]

            if len(xbars) < 81:
                continue

            if np.count_nonzero(np.isfinite(close)) < len(close) * 0.9:
                continue

            res = self._predict(xbars, threshold=threshold)
            # frame, operation, dist, profit, risk,  sample_code, sample_point, desc
            row = [tail]
            if len(res) > 0:
                yhead = tf.shift(tail, 1, frame_type)
                ytail = tf.shift(tail, 5, frame_type)

                ybars = await sec.load_bars(yhead, ytail, frame_type)
                if len(ybars) < 5 or np.any(ybars["close"] == None):
                    continue

                # operation
                row.append(res[0]["op"])
                flag = copysign(1, res[0]["op"])
                xclose = fillna(close)

                # distance
                row.append(res[0]["d"])

                # profit or gain of avoiding loss
                row.append(flag * (max(ybars["close"]) / xclose[-1] - 1))
                # risk if we act or not act
                row.append(min(ybars["low"]) / xbars["close"][-1] - 1)

                # sample_code, sample_point and desc
                row.extend([res[0]["code"], res[0]["end"], res[0]["desc"]])

            results.append(row)

        if len(results[0]) == 1:
            return None

        df = pd.DataFrame(
            results, columns=["时间", "操作", "误差", "收益", "风险", "模板", "取样点", "特征"]
        )

        return df

    def draw_test_report(
        self, df: pd.DataFrame, code: str, end: Frame, frame_type: FrameType
    ):
        opcode = {
            2: "买入",
            1: "轻仓参与",
            0: "持仓不动",
            -1: "减仓",
            -1: "清仓"
        }

        return (
            df.style.format(
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
async def train(datafile:str, version=None):
    s = SimVecStrategy("sv-30m")
    await s.train(datafile, version)

@async_run_command
async def predict(model:str, code: str, end:str=None, ft: str='30m', threshold=3e-2):
    """[summary]

    Args:
        model (str): [description]
        code (str): [description]
        end (str, optional): [description]. Defaults to None.
        ft (str, optional): [description]. Defaults to '30m'.
        threshold ([type], optional): Defaults to 3e-2. according to test, even x == y, L2 (sqrt(dot(x,x) - 2 * dot(x,y) + dot(y,y)) will still get 0.0015. so it's better to choose 3e-2 ad threshold
    """
    s = SimVecStrategy("sv-30m")

    if os.path.exists(model):
        s.load(model)
    elif model.lower().startswith("v"):
        path = os.path.expanduser(f"~/alpha/data/{s.name}/{s.name}-{model}.pkl")
        s.load(path)
    print(await s.predict(code, end, ft, threshold=threshold))

@async_run_command
async def test(model: str, code: str, n:int = 10, end:str=None, ft:str='30m', threshold=3e-3):
    s = SimVecStrategy("sv-30m")

    if os.path.exists(model):
        s.load(model)
    elif model.lower().startswith("v"):
        path = os.path.expanduser(f"~/alpha/data/{s.name}/{s.name}-{model}.pkl")
        s.load(path)

    df = await s.test(code, n, end, ft, threshold)
    print(df)


if __name__ == "__main__":
    fire.Fire({
        "train": train,
        "predict": predict,
        "test": test
    })
