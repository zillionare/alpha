import asyncio
import functools
import logging
import os
from math import copysign
from typing import List

import arrow

import cfg4py
import fire
import numpy as np
import omicron
import pandas as pd
from alpha.core.features import (
    fillna,
    moving_average,
    relation_with_prev_high,
    relationship_with_prev_low,
    relative_strength_index,
    top_n_argpos,
    volume_features,
)
from alpha.core.smvecstore import SmallSizeVectorStore
from alpha.utils import round
from numpy.typing import ArrayLike
from omicron import cache
from omicron.core.timeframe import tf
from omicron.core.types import Frame, FrameType
from omicron.models.security import Security

cfg = cfg4py.init("/apps/alpha/alpha/config")
logger = logging.getLogger(__name__)


class Twins:
    """寻找相似图形来选股的策略。"""

    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.store = SmallSizeVectorStore(
            name,
            {
                "op": "<i4",
                "code": "O",
                "end": "O",
                "desc": "O",
            },
        )
        self.nbars = kwargs.get("nbars", 81)
        self.metric = "L2"
        self.ma_wins = [5, 10, 20, 60, 120, 250]
        self.vol_win = 80

    def load(self, model: str) -> None:
        if os.path.exists(model):
            self.store = SmallSizeVectorStore.load(model)
        elif model.startswith("v"):
            path = os.path.expanduser(
                f"~/alpha/data/{self.name}/{self.name}-{model}.pkl"
            )
            self.store = SmallSizeVectorStore.load(path)

    async def add_pattern(
        self, op: int, code: str, end: str, desc: str, frame_type: str = "30m"
    ) -> None:
        sec = Security(self.canonicalize(code))
        end = arrow.get(end)
        ft = FrameType(frame_type)

        start = tf.shift(end, -self.nbars + 1, ft)
        bars = await sec.load_bars(start, end, ft)
        if (
            np.count_nonzero(bars["close"] == None) > len(bars) * 0.1
            or len(bars) < self.nbars
        ):
            return None

        vec = self.x_transform(bars)
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

    def canonicalize(self, code: str):
        if code.endswith(".XSHE") or code.endswith(".XSHG"):
            return code

        if code.startswith("6"):
            return code + ".XSHG"
        else:
            return code + ".XSHE"

    def x_transform(self, bars, flen: int = 10):
        """
        1. 均线使用5, 10, 20, 60各`flen`根,计算ma,及ma之间的发散程度。共需要66个周期的close
        2. 计算60周期以来最大的3次成交量的方向、间隔，如果成交量大于成交均量1倍
        3. 计算RSI中低于20，大于80时，距当前的距离tanh(i*2/len(bars))
        4. 成交价特征：最后10周期阴阳线特征和涨跌幅特征
        Args:
            bars ([type]): [description]
        """
        morph = []
        vec = []
        close = bars["close"]
        if np.count_nonzero(np.isfinite(close)) < len(close) * 0.9:
            return None

        close = fillna(close.copy())
        mas = []
        for win in self.ma_wins:
            ma = moving_average(close, win)[-flen:]
            ma = ma/ma[0]
            morph.extend(ma)
            mas.append(ma[-1])

        vec.extend(volume_features(bars, self.vol_win))

        # RSI
        rsi = relative_strength_index(close)[-4:]
        vec.extend(rsi)

        # 各均线的发散程度？太发散则有回归需求；太聚拢则有发散需求
        diverge = (mas[0] - mas[-1])/(mas[0] + mas[-1])
        vec.append(diverge)

        # 成交价变化特征
        vec.extend(self.price_features(close, flen))

        # 与前高的关系
        vec.extend(relation_with_prev_high(close[-60:]))

        # 与前低的关系
        vec.extend(relationship_with_prev_low(close[-60:]))

        return vec

    def price_features(self, close: np.array, flen: int = 10):
        """计算收盘价的特征

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

    def _predict(self, bars, threshold=1e-2, n: int = 1):
        """预测"""
        vec = self.normalize(self.x_transform(bars))

        if vec is None:
            return None

        res = self.store.search_vec(vec, threshold, metric=self.metric)
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
        return self.store.remove("code", code, end=arrow.get(end))

    def save(self, model: str):
        if os.path.exists(os.path.expanduser(model)):
            self.store.save(model)
        elif model.startswith("v"):
            path = os.path.expanduser(
                f"~/alpha/data/{self.name}/{self.name}-{model}.pkl"
            )
            self.store.save(path)


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