import datetime
import itertools
import logging
import pickle
import random
from typing import Callable, List, NewType, Tuple

import arrow

import numpy as np
from alpha.core.errors import NoFeaturesError, NoTargetError
from alpha.utils.data.databunch import DataBunch
from omicron import cache
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

Frame = NewType("Frame", (datetime.date, datetime.datetime))


def dataset_scope(
    start: Frame,
    end: Frame,
    codes=None,
    has_register_ipo=False,
    frame_type=FrameType.DAY,
) -> List[Tuple[str, Frame]]:
    """generate sample points for making dataset.

    Use this function to exclude duplicate points.

    Args:
        start: start frame
        end: end frame
        codes: list of securities

    Returns:
        A list of (frame, code)
    """
    secs = Securities()

    if frame_type in tf.minute_level_frames:
        convertor = tf.int2time
        timemark = arrow.get("2020-07-20 15:00+08:00")
    else:
        convertor = tf.int2date
        timemark = datetime.date(2020, 7, 20)

    if has_register_ipo:
        if codes is None:
            codes = secs.choose(_types=["stock"], exclude_st=True)

        frames = [convertor(x) for x in tf.get_frames(start, end, frame_type)]

        codes = random.sample(codes, len(codes))
        frames = random.sample(frames, len(frames))

        return itertools.product(frames, codes)

    codes_before_july = secs.choose(_types=["stock"], exclude_st=True, exclude_688=True)
    codes_after_july = secs.choose(
        _types=["stock"], exclude_st=True, exclude_688=True, exclude_300=True
    )

    if end < timemark:
        frames = [convertor(x) for x in tf.get_frames(start, end, frame_type)]
        codes = codes or codes_before_july

        codes = random.sample(codes, len(codes))
        frames = random.sample(frames, len(frames))
        return itertools.product(frames, codes)

    frames1 = [convertor(x) for x in tf.get_frames(start, timemark, frame_type)]

    codes = codes or codes_before_july
    codes = random.sample(codes, len(codes))
    frames1 = random.sample(frames1, len(frames1))
    permutations1 = itertools.product(frames1, codes)

    frames2 = [convertor(x) for x in tf.get_frames(timemark, end, frame_type)]

    codes = codes or codes_after_july
    frames2 = random.sample(frames2, len(frames2))
    permutations2 = itertools.product(frames2, codes)

    return itertools.chain(permutations1, permutations2)


async def make_dataset(
    transformers: dict,
    target_transformer: Callable,
    target_win: int,
    total: int,
    nbuckets: int,
    secs: List[str] = None,
    start: Frame = None,
    end: Frame = None,
    main_frame=FrameType.DAY,
    has_register_ipo=False,
    notes:str=None,
    epoch = 100
) -> DataBunch:
    """生成数据集

    本方法驱动transformers来提取特征，并将其转化为DataBunch。

    `transformers` 定义为一个集合： {
        frame_type: {
            name: name of the transformer, optional,
            func: transformer function,
            bars_len: length of bars, used by `make_dataset` to load bars
            }
        }.

    `bars_len` defined how many bars to be load for the `frame_type` which will be used by by `func`. The end frame of the bars will be end bars for `main_frame`.

    transformer is a function that takes bars as input and return a list of row features. In case of no feature extraction, return None/raise NoFeatureError and do not add the row to dataset.

    方法首先定义一个样本空间并遍历该空间。每次遍历时，取得一个代码和bars结束时间的组合，然后根据bars_len定义的bars长度来获取main frame type的bars。在提取main features之后，再遍历集全，处理其它周期(frame_type)。处理其它周期时，都使用main bars的最后一根bar作为结束bar的时间，然后根据该frame_type的bars长度来获取side bars。

    Args:
        transformers: a dict of {frame_type: {name: name of the transformer, optional, func: transformer function, bars_len: length of bars, used by `make_dataset` to load bars}}
        target_transformer: a function that takes bars as input and return a list of row features.
        target_win: the length of target window
        secs: list of securities
        start: start frame
        end: end frame
        main_frame: main frame type
        has_register_ipo: whether to include stock that is listed under register ipo
        notes: notes for this dataset
        epoch: number of epochs to wait before stop iterating, if `total` is not met
    Returns:
        a DataBunch
    """
    main_bars_len = transformers.get(main_frame, {}).get("bars_len", 300)

    if end is None:
        end = tf.shift(arrow.now(), -target_win - 2, main_frame)

    start = start or datetime.date(2015, 10, 9)

    buckets = [0] * nbuckets
    capacity = int(total / nbuckets) + 1

    X, y, raw = [], [], []

    permutations = dataset_scope(start, end, secs, has_register_ipo)

    main_transformer = transformers.get(main_frame, {}).get("func")
    assert main_transformer is not None, "main_transformer is None"

    for i, (tail, code) in enumerate(permutations):
        sec = Security(code)

        head = tf.shift(tail, -main_bars_len + 1, main_frame)

        # ensure we got bars from cache only
        _head, _tail = await cache.get_bars_range(code, FrameType.DAY)
        if _head is None or _tail is None or _head >= head or _tail <= tail:
            continue

        bars = await sec.load_bars(head, tail, main_frame)

        if len(bars) != main_bars_len:
            continue

        xbars = bars[:-target_win]
        ybars = bars[-target_win:]

        if np.count_nonzero(np.isfinite(xbars["close"])) < len(xbars) * 0.95 or np.count_nonzero(np.isfinite(ybars["close"])) < len(ybars):
            continue

        side_bar_end = xbars["frame"][-1]

        # get the target value
        try:
            y_, bucket_idx = target_transformer(ybars, xbars)
            if buckets[bucket_idx] >= capacity:
                continue
        except NoTargetError:
            continue
        except Exception as e:
            logger.exception(e)
            continue

        try:
            # extract features for main frame type
            x_ = main_transformer(xbars)

            # extract features for side frame type
            for frame_type, item in transformers.items():
                if frame_type == main_frame:
                    continue

                transformer = item.get("func")
                side_bars_len = item.get("bars_len")

                if side_bars_len is None or transformer is None:
                    raise ValueError("side_bars_len, transformer are all required")

                side_bar_tail = tf.combine_time(side_bar_end, hour=15)
                side_bar_head = tf.shift(side_bar_tail, -side_bars_len, frame_type)
                side_bars = await sec.load_bars(
                    side_bar_head, side_bar_tail, frame_type
                )

                if (
                    np.count_nonzero(np.isfinite(side_bars["close"]))
                    < side_bars_len * 0.9
                ):
                    raise NoFeaturesError("not enough data")

                side_features = transformer(side_bars)

                x_.extend(side_features)
        except NoFeaturesError:
            continue
        except Exception as e:
            logger.exception(e)
            continue

        X.append(x_)
        y.append(y_)
        raw.append((code, xbars, ybars))
        buckets[bucket_idx] += 1

        if len(X) % (total // 100) == 0:
            logger.info("%s samples collected: %s", len(X), buckets)

        if len(X) >= total * 0.95:
            logger.info("have collected more than 0.95 times of required samples")
            break

        if i >= total * epoch:
            logger.info("have searched more than %s times of total space, stopped", epoch)
            break

    ds = DataBunch(X, y, raw, desc=notes)
    logger.info("%s features made", ds.X.shape)
    return ds


async def even_distributed_dataset(
    total: int,
    buckets_size: int,
    bars_len: int,
    target_to_bucket: Callable,
    save_to: str,
    has_register_ipo=False,
    start="2010-01-01",
    meta: dict = {},
    frame_type: FrameType = FrameType.DAY,
):
    """构建按涨跌幅均匀分布的数据集

    数据集的target值、以及如何映射到buket，由`target_to_bucket`来决定。

    如果遇到证券停牌的情况，则停牌时间不能超过`bars_len`的10%，否则会被丢弃。

    Args:
        total: 总数
        buckets_size: 桶个数
        bars_len: 每个桶的bars长度
        target_to_bucket: 将目标值映射到桶的函数
        has_register_ipo: 是否包含注册制股票
    Returns:
        a DataBunch
    """
    data = []

    buckets = [0] * buckets_size
    capacity = int(total / buckets_size) + 1

    start = tf.shift(arrow.get(start), 0, frame_type)
    end_date = tf.day_shift(arrow.now(), 0)
    end = tf.shift(tf.combine_time(end_date, hour=15), 0, frame_type)

    for i, (tail, code) in enumerate(dataset_scope(start, end, frame_type=frame_type)):
        sec = Security(code)
        head = tf.shift(tail, -bars_len + 1, frame_type)

        _head, _tail = await cache.get_bars_range(code, frame_type)
        if _head is None or _tail is None or _head >= head or _tail <= tail:
            continue

        bars = await sec.load_bars(head, tail, frame_type)
        if len(bars) != bars_len:
            continue

        close = bars["close"]
        if np.count_nonzero(np.isfinite(close)) < bars_len * 0.9:
            continue

        target, bucket_idx = target_to_bucket(bars)
        if target is None:
            continue

        try:
            if buckets[bucket_idx] >= capacity:
                continue
        except IndexError:
            continue

        buckets[bucket_idx] += 1
        data.append((code, target, bars))

        if len(data) >= total * 0.95 or i >= total * 100:
            break

        if len(data) % int(total / 100) == 0:
            logger.info("%s samples collected: %s", len(data), buckets)

    meta.update(
        {
            "buckets_size": buckets_size,
            "bars_len": bars_len,
            "capacity": capacity,
            "total": total,
            "start": start,
            "has_register_ipo": has_register_ipo,
        }
    )
    with open(save_to, "wb") as f:
        pickle.dump({"data": data, "meta": meta}, f)


def load_data(dataset_path: str) -> "DataBunch":
    """
    Loads data from a given dataset path.
    """
    with open(dataset_path, "rb") as f:
        bunch = pickle.load(f)
        bunch.path = dataset_path

    if bunch.name is None:
        bunch.name = dataset_path.split("/")[-1].split(".")[0]

    if bunch.desc is None:
        bunch.desc = bunch.name

    return bunch
