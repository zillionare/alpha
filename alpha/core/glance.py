"""市场环境概览
"""
from omicron.core.types import Frame, FrameType
from omicron.models.security import Security
from alpha.core.rsi_stats import RsiStats
from omicron.core.triggers import FrameTrigger
import arrow
from omicron.core.timeframe import tf
from alpha.core.features import relative_strength_index


class Glance:
    xshg = Security("000001.XSHG")

    def __init__(self):
        self.rsi_day = RsiStats("ris", Frame.DAY)
        self.rsi_30m = RsiStats("ris", Frame.MIN30)

        self.rsi_day.load()
        self.rsi_30m.load()

        self.glance = []

    def register_backend_jobs(self, scheduler):
        trigger = FrameTrigger(Frame.MIN30, "1s")
        scheduler.add_job(self.update_stats, trigger, args=(Frame.MIN30,))

        trigger = FrameTrigger(Frame.DAY, "1h")
        scheduler.add_job(self.update_stats, trigger, args=(Frame.DAY,))

    async def update_stats(self, frame_type):
        if frame_type == Frame.MIN30:
            # always calc for 1000 frames
            end = tf.floor(arrow.now(), frame_type)
            start = tf.shift(end, -1000, frame_type)

            if not (
                start == self.rsi_30m.time_range[0]
                and end == self.rsi_30m.time_range[1]
            ):
                await self.rsi_30m.calc(frame_type)
        elif frame_type == Frame.DAY:
            # always calc for 1000 frames
            end = tf.floor(arrow.now(), frame_type)
            start = tf.shift(end, -1000, frame_type)

            if not (
                start == self.rsi_day.time_range[0]
                and end == self.rsi_day.time_range[1]
            ):
                await self.rsi_day.calc(frame_type)

    async def update_status(self):
        """报告当前状态"""
        status = {}

        # to check if reach RSI top/bottom
        end = tf.floor(arrow.now(), Frame.MIN30)
        start = tf.shift(end, -40, Frame.MIN30)

        shbars = await self.xshg.load_bars(start, end, FrameType.MIN30)
        close = shbars["close"]
        rsi = relative_strength_index(close, 6)
        p = self.rsi_30m.get_proba(self.xshg.code, rsi)

        status["rsi"] = rsi
        status["prsi"] = p
