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

    async def calc(self, start, end):
        hist = {}

        secs = Securities()
        codes = secs.choose(["stock"])
        codes.extend(["000001.XSHG", "399001.XSHE", "399006.XSHE"])

        start = tf.floor(arrow.get(start), self.frame_type)
        end = tf.floor(arrow.get(end), self.frame_type)

        # this allow the caller just pass `start` and `end` in date unit, for simplicity
        if self.frame_type in tf.minute_level_frames:
            start = tf.combine_time(start, hour=15)
            end = tf.combine_time(end, hour=15)

        nbars = tf.count_frames(start, end, self.frame_type)

        for code in codes:
            try:
                sec = Security(code)
                bars = await sec.load_bars(start, end, self.frame_type)

                if len(bars) < 0.75 * nbars:
                    logger.info(
                        f"{sec.display_name} contains no enough data from {start} to {end}, skip calculating."
                    )
                    continue

                close = fillna(bars["close"].copy())
                rsi = relative_strength_index(close, period=6)
                hist[code] = np.histogram(rsi, bins=100, range=(0, 100))

            except Exception:
                continue

        file = f"~/zillionare/alpha/stats/rsi_{self.frame_type.value}.pkl"
        with open(os.path.expanduser(file), "wb") as f:
            pickle.dump({"time_range": [start, end], "hist": hist}, f)

    def get_proba(self, code, value):
        """given `value`, find relative continuouse density probability"""
        if code in self.cdfs:
            return self.cdfs.get(code).cdf(value)
        return None

    def get_value(self, code, proba):
        """given probability, find the corresponding value"""
        if code in self.cdfs:
            return self.cdfs.get(code).ppf(proba)
        return None


rsi30 = RsiStats(FrameType.MIN30)
rsiday = RsiStats(FrameType.DAY)
rsi30.load()
rsiday.load()

__all__ = ["rsi30", "rsiday"]
