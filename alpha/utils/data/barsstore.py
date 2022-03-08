from collections import defaultdict
import datetime
import logging
import os
import pickle
from typing import Any

import arrow
import jqdatasdk as jq
import numpy as np
import omicron
from coretypes import FrameType, SecurityType
from omicron.models.timeframe import TimeFrame as tf

logger = logging.getLogger(__name__)


class NoEnoughQuotaError(Exception):
    pass


class BarsStore:
    """Store bars in this way:

    {
        "code": bars # np.ndarray
    }

    one store only keep bars for one security type + one frame type
    """

    def __init__(
        self,
        path: str,
        ft: FrameType = None,
        sec_type: SecurityType = None,
        epoch_start="2019-01-04",
    ):
        """

        Args:
            filepath : 文件路径, 格式为 f"{sec_type.value}.{ft.value}.pkl"
            ft : _description_.
            sec_type : _description_.
        """
        if ft not in [*tf.minute_level_frames, FrameType.DAY]:
            raise ValueError(f"ft={ft} not support")

        self.filepath = os.path.join(path, f"{sec_type.value}.{ft.value}.pkl")

        self.data = {}
        self.ft = ft
        self.sec_type = sec_type
        self.start = None
        self.end = None

        self.epoch_start = arrow.get(epoch_start).date()

        if os.path.exists(self.filepath):
            self.load()

    def load(self):
        with open(self.filepath, "rb") as f:
            data = pickle.load(f)

            if self.ft is not None:
                assert data["ft"] == self.ft, "ft not match"

            if self.sec_type is not None:
                assert data["sec_type"] == self.sec_type, "sec_type not match"

            self.ft = data["ft"]
            self.sec_type = data["sec_type"]
            self.data = data["bars"]
            self.start = data["start"]
            self.end = data["end"]

    def save(self):
        assert self.ft is not None, "ft not set"
        assert self.sec_type is not None, "sec_type not set"
        assert self.start is not None, "start not set"
        assert self.end is not None, "end not set"

        self.data = {
            code: np.concatenate(bars, axis=0) for code, bars in self._for_sync.items()
        }
        with open(self.filepath, "wb") as f:
            content = {
                "ft": self.ft,
                "sec_type": self.sec_type,
                "start": self.start,
                "end": self.end,
                "bars": self.data,
            }

            pickle.dump(content, f)

    def __getitem__(self, key):
        return self.data[key]

    def push_right(self, code: str, bars: np.ndarray):
        """右侧插入bars。

        如果对数据有排序要求（通常情况下是），需要先排序，然后再插入。
        """
        if code not in self._for_sync:
            self._for_sync[code] = [bars]
        else:
            self._for_sync[code].append(bars)

    def push_left(self, code: str, bars: np.ndarray):
        """左侧插入bars。"""
        if code not in self._for_sync:
            self._for_sync[code] = [bars]
        else:
            self._for_sync[code].insert(0, bars)

    def next_sync_date(self):
        """返回下一个需要同步的日期。

        如果没有数据，返回None。
        """
        if any([self.start is None, self.end is None]):
            self.start = self.end = None

            return tf.day_shift(arrow.now(), 0)
        else:
            assert all(
                [self.start is not None, self.end is not None]
            ), "start and end must all exist"

            now = arrow.now()
            next_trade_day = tf.day_shift(now, 0)
            if next_trade_day > self.end:
                return next_trade_day
            else:
                if self.start <= self.epoch_start:
                    return None
                return tf.day_shift(self.start, -1)

    def sync_bars(self):
        sync_date = self.next_sync_date()

        if self.ft == FrameType.DAY:
            bars_per_day = 1
        elif self.ft == FrameType.MIN30:
            bars_per_day = 8
        elif self.ft == FrameType.MIN15:
            bars_per_day = 16
        elif self.ft == FrameType.MIN5:
            bars_per_day = 48
        elif self.ft == FrameType.MIN1:
            bars_per_day = 240
        else:
            raise ValueError(f"sync for {self.ft} not support")

        # temporary data structure for sync only
        self._for_sync = {code: [bars] for code, bars in self.data.items()}

        while sync_date is not None:
            if sync_date < self.epoch_start:
                break

            logger.info("sync bars on date %s", sync_date)
            try:
                self.sync_bars_on_date(sync_date, bars_per_day)
            except NoEnoughQuotaError:
                logger.info("sync failed, no enough quota")
                break
            except Exception as e:
                logger.exception(e)

            sync_date = self.next_sync_date()

        self.save()
        self._for_sync = None

    def sync_bars_on_date(self, sync_date: datetime.date, bars_per_day: int):
        """同步数据，把数据库中的数据插入到当前对象中。

        如果forward为真，则数据追加到当前数据的右侧; 否则，数据追加到当前数据的左侧。
        """
        quota = jq.get_query_count()
        spare = quota.get("spare", 0)

        MAX_RESULT_SIZE = 3000

        try:
            secs = (
                jq.get_all_securities(types=[self.sec_type.value], date=sync_date)
            ).index.tolist()
        except Exception as e:
            if str(e).find("最大查询限制") != -1:
                raise NoEnoughQuotaError(f"spare is {spare}")

        recs_to_sync = len(secs) * bars_per_day
        if recs_to_sync > spare:
            raise NoEnoughQuotaError(f"Aborted: {recs_to_sync}, only {spare} available")

        batches = recs_to_sync // MAX_RESULT_SIZE + 1
        batch_size = recs_to_sync // batches

        end = datetime.datetime.combine(sync_date, datetime.time(15))

        fields = ["date", "open", "high", "low", "close", "volume", "money", "factor"]

        if self.end and sync_date > self.end:
            append = self.push_right
        else:
            append = self.push_left

        for i in range(batches):
            codes = secs[i * batch_size : (i + 1) * batch_size]
            data = jq.get_bars(
                codes,
                bars_per_day,
                self.ft.value,
                include_now=True,
                fields=fields,
                end_dt=end,
                df=False,
            )

            for code, bars in data.items():
                # jqdata有可能返回前一日的数据，如果当日停牌的话
                if self.ft == FrameType.DAY:
                    if bars[0]["date"] != end.date():
                        continue
                else:
                    if bars[0]["date"] != end.date():
                        continue
                append(code, bars)

        if self.end is None:
            self.start = self.end = sync_date
        elif sync_date > self.end:
            self.end = sync_date
        elif sync_date < self.start:
            self.start = sync_date


class LofStore(BarsStore):
    def __init__(self, path: str, ft: FrameType, epoch_start: str):
        super().__init__(path, ft, SecurityType.LOF, epoch_start)


if __name__ == "__main__":
    import asyncio

    import cfg4py

    from alpha.config import get_config_dir

    cfg = cfg4py.init(get_config_dir())

    asyncio.run(omicron.init())

    account = os.environ.get("JQ_ACCOUNT")
    password = os.environ.get("JQ_PASSWORD")

    jq.auth(account, password)

    lof = LofStore("/tmp", FrameType.DAY, "2015-01-01")
    lof.sync_bars()
