from alpha.core.features import fillna, relative_strength_index
from omicron.core.types import FrameType
import os
import pickle
from scipy.stats import rv_histogram
from omicron.models.security import Security
from omicron.models.securities import Securities
import arrow
from omicron.core.timeframe import tf
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RsiStats:
    def __init__(self, frame_type: FrameType):
        self.cdfs = {}
        self.frame_type = frame_type
        self.time_range = []

    def load(self):
        """pickled data contains time_range and cdfs

        time_range indicates the time range that the indicator is calculated
        """
        file_path = f"~/zillionare/alpha/stats/rsi_{self.frame_type.value}.pkl"
        file_path = os.path.expanduser(file_path)

        if not os.path.exists(file_path):
            return False

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        self.time_range = data["time_range"]
        hist = data["hist"]
        for code, values in hist.items():
            self.cdfs[code] = rv_histogram(values)

    async def calc(self, end=None):
        hist = {}

        secs = Securities()
        codes = secs.choose(["stock"])
        codes.extend(["000001.XSHG", "399001.XSHE", "399006.XSHE"])

        if end is None:
            end = arrow.now()

        end = tf.floor(arrow.get(end), self.frame_type)
        start = tf.shift(end, -1000 - 18 + 1, self.frame_type)

        # this allow the caller just pass `start` and `end` in date unit, for simplicity
        if self.frame_type in tf.minute_level_frames:
            start = tf.combine_time(start, hour=15)
            end = tf.combine_time(end, hour=15)

        nbars = tf.count_frames(start, end, self.frame_type)
        logger.info("calc rsi from latest %s nbars", nbars)

        for code in codes:
            try:
                sec = Security(code)
                bars = await sec.load_bars(start, end, self.frame_type)
                close = bars["close"]
                valid_bars = np.count_nonzero(np.isfinite(close))
                if valid_bars < nbars * 0.75:
                    logger.warning(
                        "%s contains only %s bars, less than required", code, valid_bars
                    )
                    continue

                close = fillna(bars["close"].copy())
                # 前18位均为nan
                rsi = relative_strength_index(close, period=6)[18:]
                hist[code] = np.histogram(rsi, bins=100, range=(0, 100))

            except Exception:
                continue

        file = f"~/zillionare/alpha/stats/rsi_{self.frame_type.value}.pkl"
        with open(os.path.expanduser(file), "wb") as f:
            pickle.dump({"time_range": [start, end], "hist": hist}, f)

    def get_proba(self, code, value):
        """given `value`, find relative continuouse density probability"""
        if code in self.cdfs:
            return round(self.cdfs.get(code).cdf(value), 2)
        return None

    def get_rsi(self, code, proba):
        """given probability, find the corresponding RSI"""
        if code in self.cdfs:
            return self.cdfs.get(code).ppf(proba)
        return None


rsi30 = RsiStats(FrameType.MIN30)
rsiday = RsiStats(FrameType.DAY)
rsi30.load()
rsiday.load()

__all__ = ["rsi30", "rsiday"]
