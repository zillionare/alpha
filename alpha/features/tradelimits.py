import datetime
import logging

import arrow
from coretypes import FrameType
from omicron.models.stock import Stock
from omicron.models.timeframe import TimeFrame as tf

logger = logging.getLogger(__name__)


async def trade_limits(code: str, n: int, end: datetime.date = None):
    if end is None:
        end = arrow.now().date()

    bars = await Stock.get_bars(code, n, FrameType.DAY, end, fq=False)
    if len(bars) == 0:
        return None

    start = tf.day_shift(end, -n + 1)
    limits = await Stock.get_trade_price_limits(code, start, end)

    if limits.size != bars.size:
        logger.warning(
            "data error: size not equal for %s from %s to %s", code, start, end
        )
        return None

    reach_high = abs(bars["close"] - limits["high_limit"]) < 1e-2
    reach_low = abs(bars["close"] - limits["low_limit"]) < 1e-2

    return reach_high, reach_low
