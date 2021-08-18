import datetime
import functools
import imp
from alpha.core.features import fillna, moving_average, relative_strength_index, top_n_argpos
from omicron.core.types import Frame, FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security
from omicron.core.timeframe import tf
from alpha.utils import round
import arrow
import numpy as np
from numpy.typing import ArrayLike
import asyncio
import omicron
import cfg4py
from alpha.config import get_config_dir
import fire
import logging

from pymilvus import Milvus, DataType

logger = logging.getLogger(__name__)
milvus = Milvus("172.17.0.1", "19530")


def async_run_command(func):
    async def _init_and_run(*args, **kwargs):
        cfg4py.init("/apps/alpha/alpha/config")
        await omicron.init()

        await func(*args, **kwargs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        asyncio.run(_init_and_run(*args, **kwargs))

    return wrapper


def init_stocks_collection(dim: int, drop=False):
    id_field = {
        "name": "id",
        "type": DataType.INT64,
        "auto_id": True,
        "is_primary": True,
    }

    code_field = {"name": "code", "type": DataType.INT32}

    start_field = {"name": "start", "type": DataType.INT32}

    feature_field = {
        "name": "features",
        "type": DataType.FLOAT_VECTOR,
        "metric_type": "L2",
        "params": {"dim": dim},
        "indexes": [{"metric_type": "L2"}],
    }

    fields = {"fields": [id_field, code_field, start_field, feature_field]}
    if milvus.has_collection("stock") and drop:
        milvus.drop_collection("stock")
    elif not milvus.has_collection("stock"):
        milvus.create_collection("stock", fields)


async def predict():
    milvus.load_collection("stock")

    vec = None

    topK = 5
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    code = "000004.XSHE"
    sec = Security(code)
    start = datetime.date(2018, 1, 1)
    end = datetime.date(2021, 8, 1)

    bars = await sec.load_bars(start, end, FrameType.DAY)

    for i in range(30, len(bars)):
        close = bars["close"][i - 30 : i]
        if np.count_nonzero(np.isfinite(close)) < len(close) * 0.9:
            continue

        vec = []
        close = fillna(close.copy())
        vec.extend(close / close[0])

        volume = bars["volume"][i - 30 : i]
        vec.extend(volume / volume[0])

        res = milvus.search_with_expression(
            "stock",
            [vec],
            "features",
            param=search_params,
            limit=2,
            output_fields=["code", "start"],
        )

        print(res)


def ma_features(close: np.array, wins=(5, 10, 20, 60), flen: int = 7):
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


def volume_features(volume: np.array, flags: ArrayLike, win: int = 80):
    """计算`win`周期以来，最大3次成交量的方向和间隔，如果成交量大于成交均量的1倍

    Args:
        volume (np.array): [description]
        flags (ArrayLike): 当前成交量的方向
        win (int, optional): [description]. Defaults to 80.

    Raises:
        ValueError: [description]
    """
    if len(volume) < win or len(flags) < win:
        raise ValueError("len of volume/flags should be >= 60")

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
    np.pad(indice, (0, 3 - len(indice)), 'constant', constant_values=0)
    vec.extend(flags[indice])


    # 间隔
    vec.extend([np.tanh((win - i) / (win/2)) for i in indice])
    return vec

def canonicalize(code:str):
    if code.startswith("6"):
        return code + ".XSHG"
    else:
        return code + ".XSHE"

def xtransform(bars, flen:int = 7):
    """
    1. 均线使用5, 10, 20, 60各7根,计算ma,及ma之间的发散程度。共需要66个周期的close
    2. 计算60周期以来最大的3次成交量的方向、间隔，如果成交量大于成交均量1倍
    3. 计算RSI中低于20，大于80时，距当前的距离tanh(i/14)
    4. 成交价特征：最后10周期阴阳线特征和涨跌幅特征
    Args:
        bars ([type]): [description]
    """
    vec = []
    close = bars["close"]
    if np.count_nonzero(np.isfinite(close)) < len(close) * 0.9:
        return None

    close = fillna(close.copy())
    vec.extend(ma_features(close))

    volume = fillna(bars["volume"].copy())
    vec.extend(volume_features(volume, close))

    # RSI
    rsi = relative_strength_index(close, 6)
    last_min = np.argmax(rsi < 0.2)
    last_max = np.argmax(rsi > 0.8)




async def train(nbars:int = 80, frame_type:FrameType = FrameType.MIN30):
    samples = [
        (1, "300985", "20210817 10:00", "高开低走，放量大阴，RSI反转信号已出，滞胀"),
        (-1, "300985", "20210809 15:00", "放量涨、缩量跌，均线多头，m5,m10,m20收敛，m60向上支撑"),
    ]

    recs = []
    for (flag, code, end, desc) in samples:
        sec = Security(canonicalize(code))

        if isinstance(end, str):
            end = arrow.get(end)

        start = tf.shift(end, -nbars, frame_type)
        bars = await sec.load_bars(start, end, frame_type)

        vec = xtransform(bars)
        if vec is None:
            continue

        icode = int("1" + code.split(".")[0])
        idate = tf.date2int(end)
        recs.append([{
            "name": "code",
            "type": DataType.INT32,
            "value": [icode],
        }, {
            "name": "end" ,
            "type": DataType.INT32,
            "value": [idate]
        }, {
            "name": "flag",
            "type": DataType.INT32,
            "value": [flag]
        },{
            "name": "features",
            "type": DataType.FLOAT_VECTOR,
            "value": vec
        }])

    dim = len(recs[0][-1]["value"][0])
    init_stocks_collection(dim, drop = True)
    ids = milvus.insert("stock", recs)
    logger.info("%s samples inserted into milvus", len(ids))



async def pred():
    pass


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "pred": pred,
        }
    )
