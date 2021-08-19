from typing import Callable
import datetime
from dis import dis
import functools
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
import fire
import logging
from sklearn.preprocessing import normalize

from pymilvus import Milvus, DataType

logger = logging.getLogger(__name__)
#milvus = Milvus("172.17.0.1", "19530")
milvus = Milvus("172.17.22.1", "19530")


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
    end_field = {"name": "end", "type": DataType.INT64}
    flag_field = {"name": "flag", "type": DataType.INT32}

    feature_field = {
        "name": "features",
        "type": DataType.FLOAT_VECTOR,
        "metric_type": "L2",
        "params": {"dim": dim},
        "indexes": [{"metric_type": "L2"}],
    }

    fields = {"fields": [id_field, code_field, end_field, flag_field, feature_field]}
    if milvus.has_collection("stock") and drop:
        milvus.drop_collection("stock")
    
    if not milvus.has_collection("stock"):
        milvus.create_collection("stock", fields)

def _predict(bars, threshold:float=3e-3, n:int=1):
    milvus.load_collection("stock")
    vecs = normalize([xtransform(bars)]).tolist()

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    res = milvus.search_with_expression(
            "stock",
            vecs,
            "features",
            param=search_params,
            limit=1,
            output_fields=["code", "end", "flag"],
        )
    
    results = []
    for item in res[0]:
        d = item.distance
        if d < threshold:
            code = str(item.entity.get('code'))[1:]
            end = tf.int2time(item.entity.get("end"))
            flag = item.entity.get("flag")

            results.append((code, end, flag, d))

    return results

    
@async_run_command
async def predict(code:str, end:str, frame_type:str="30m"):
    code = canonicalize(code)
    end = arrow.get(end)
    frame_type = FrameType(frame_type)

    sec = Security(code)
    start = tf.shift(end, -80, frame_type)

    bars = await sec.load_bars(start, end, frame_type)

    results = _predict(bars)
    if len(results) > 0:
        *_, flag, distance = results[0]
        print(flag, f"{distance:.3f}")
    else:
        print(f"no match patterns")

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
    np.pad(indice, (0, 3 - len(indice)), 'constant', constant_values=0)
    vec.extend(flags[indice])


    # 间隔
    vec.extend([np.tanh( 2 * (win - i - 1) / win) for i in indice])
    return vec

def rsi_features(close: np.array, rsi_win: int = 6):
    rsi = relative_strength_index(close, rsi_win)

    vec = []
    nbars = len(close)

    min_rsi_pos = np.argwhere(rsi < 20)
    if len(min_rsi_pos) > 0:
        min_rsi_pos = min_rsi_pos[-1] + rsi_win
        vec.extend(np.tanh((nbars -1 - min_rsi_pos) * 2/ nbars))
    else:
        vec.append(1)

    max_rsi_pos = np.argwhere(rsi > 80)
    if len(max_rsi_pos) > 0:
        max_rsi_pos = max_rsi_pos[-1] + rsi_win
        vec.extend(np.tanh((nbars - 1 - max_rsi_pos) * 2 / nbars))
    else:
        vec.append(1)

    return vec
    
def canonicalize(code:str):
    if code.endswith(".XSHE") or code.endswith(".XSHG"):
        return code

    if code.startswith("6"):
        return code + ".XSHG"
    else:
        return code + ".XSHE"

def xtransform(bars, flen:int = 7):
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
    vec.extend(ma_features(close))

    volume = fillna(bars["volume"].copy())
    vec.extend(volume_features(volume, np.where(close[1:] > close[:-1], 1, -1)))

    # RSI
    vec.extend(rsi_features(close))

    # 成交价变化特征
    vec.extend((close[1:]/close[:-1])[-flen:])

    return vec

def _train(sample_bars: ArrayLike, time_convertor: Callable):
    codes, dates, flags, features = [], [], [], []

    for (flag, code, end, bars) in sample_bars:
        vec = xtransform(bars)

        if vec is None:
            continue

        codes.append(int("1" + code.split(".")[0]))
        dates.append(time_convertor(end))
        flags.append(flag)
        features.append(vec)

    dim = len(vec)
    init_stocks_collection(dim, drop = True)

    features = normalize(features).tolist()
    logger.debug("\n%s", features)
    mr = milvus.insert("stock", [{
            "name": "code",
            "type": DataType.INT32,
            "values": codes,
        }, {
            "name": "end" ,
            "type": DataType.INT64,
            "values": dates
        }, {
            "name": "flag",
            "type": DataType.INT32,
            "values": flags
        },{
            "name": "features",
            "type": DataType.FLOAT_VECTOR,
            "values": features
        }])
    logger.info("%s samples inserted into milvus", len(mr.primary_keys))


@async_run_command
async def train(nbars:int = 80, frame_type:FrameType = FrameType.MIN30):
    samples = [
        (-1, "300985", "20210817 10:00", "高开低走，放量大阴，资金方向为卖出，RSI反转信号已出，滞胀，此时卖出为最佳时机"),
        (-1, "300985", "20210817 11:30", "m5均线已拐头，上攻被均线压制，资金方向为卖出。应及时止损"),
        (1, "300985", "20210803 11:30", "均线多头，最能萎缩，m5均线再度小幅发散，起涨前夜。"),
        (1, "300985", "20210809 15:00", "放量涨、缩量跌，均线多头，m5,m10,m20收敛，m60向上支撑"),
        (1, "300985", "20210816 10:00", "均线多头，资金方向为买入。但m5,m10,m20已高度发散，可能进入了最后拉升周期"),
    ]

    sample_bars = []
    for (flag, code, end, desc) in samples:
        sec = Security(canonicalize(code))

        if isinstance(end, str):
            end = arrow.get(end)

        start = tf.shift(end, -nbars, frame_type)
        bars = await sec.load_bars(start, end, frame_type)
        sample_bars.append((flag, code, end, bars))

    convertor = tf.time2int if frame_type in tf.minute_level_frames else tf.date2int
    _train(sample_bars, convertor)

def show_status():
    milvus.load_collection("stock")
    print(milvus.get_collection_stats("stock"))

@async_run_command
async def test(code:str, start: str, n:int=10, frame_type:str='30m'):
    """对`code`在`start`指定的`n`个周期里，进行pattern匹配测试。
    Args:
        code (str): 股票代码
        start (str): 开始时间
        n (int): 周期数
        frame_type (str): 周期类型
    """
    code = str(code)
    sec = Security(canonicalize(code))
    frame_type = FrameType(frame_type)
    start = arrow.get(start)
    end = tf.shift(start, n, frame_type)

    convertor = tf.int2time if frame_type in tf.minute_level_frames else tf.int2date
    for end in  tf.get_frames(start, end, frame_type):
        end = convertor(end)
        start = tf.shift(end, -80, frame_type)
        bars = await sec.load_bars(start, end, frame_type)
        if len(bars) < 81:
            continue

        result = _predict(bars)
        if len(result) > 0:
            *_, flag, distance = result[0]
            print(f"{end}\t{flag}\t{distance:.3f}")
        else:
            print(f"{end}\t{0}\tNA")



if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "pred": predict,
            "test": test,
            "status": show_status,
        }
    )
