from alpha.core.errors import NoFeaturesError, NoTargetError
from typing import Callable, NewType
from typing import List
from typing import Tuple
import arrow
from omicron.models.securities import Securities
from omicron.models.security import Security
import itertools
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
import random
import datetime
from alpha.utils.data.databunch import DataBunch
import numpy as np

import logging

logger = logging.getLogger(__name__)

Frame = NewType("Frame", (datetime.date, datetime.datetime))


def dataset_scope(start: Frame, end: Frame, codes=None) -> List[Tuple[str, Frame]]:
    """generate sample points for making dataset.

    Use this function to exclude duplicate points.

    Args:
        start: start frame
        end: end frame
        codes: list of securities

    Returns:
        A list of (frame, value)
    """
    if codes is None:
        secs = Securities()
        codes = secs.choose(_types=["stock"])
    frames = [tf.int2date(x) for x in tf.get_frames(start, end, FrameType.DAY)]

    permutations = list(itertools.product(codes, frames))

    logger.info("%s permutaions in total", len(codes) * len(frames))
    return random.sample(permutations, len(codes) * len(frames))


async def make_dataset(
    transformers: dict,
    target_transformer: Callable,
    target_win: int,
    bucket: Tuple[int, int, Callable],
    secs: List[str] = None,
    start: Frame = None,
    end: Frame = None,
    main_frame=FrameType.DAY,
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
        bucket: a tuple of (start, end, func)
        secs: list of securities
        start: start frame
        end: end frame
        main_frame: main frame type
    Returns:
        a DataBunch
    """
    if secs is None:
        secs = Securities()
        codes = secs.choose(_types=["stock"])

    main_bars_len = transformers.get(main_frame, {}).get("bars_len", 300)

    if end is None:
        end = tf.shift(arrow.now(), -target_win - 2, main_frame)

    bucket_size, capacity, check_full = bucket
    buckets = [0] * bucket_size
    if start is None:
        samples_per_sec = int(bucket_size * capacity * 100 / len(codes))

        # for able to collect enough data, we need to loose the scope
        start = tf.shift(end, -samples_per_sec * 10, main_frame)

    X, y, raw = [], [], []

    permutations = dataset_scope(start, end, codes)

    main_transformer = transformers.get(main_frame, {}).get("func")
    assert main_transformer is not None, "main_transformer is None"

    for code, tail in permutations:
        sec = Security(code)

        head = tf.shift(tail, -main_bars_len + 1, main_frame)
        bars = await sec.load_bars(head, tail, main_frame)

        if len(bars) != main_bars_len:
            continue

        xbars = bars[:-target_win]
        ybars = bars[-target_win:]

        side_bar_end = xbars["frame"][-1]

        # get the target value
        try:
            y_ = target_transformer(ybars, xbars)
        except NoTargetError:
            continue

        if check_full(buckets, y_, capacity):
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

        if len(X) % 100 == 0:
            logger.info("%s samples collected: %s", len(X), buckets)

        if sum(buckets) >= capacity * 0.85 * bucket_size:
            break

    ds = DataBunch(X, y, raw)
    logger.info("%s features made", ds.X.shape)
    return ds
