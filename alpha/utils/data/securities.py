import datetime
import logging
import os
from typing import List

import arrow
import fire
import jqdatasdk as jq
import numpy as np
import pandas as pd
import zarr
from coretypes import FrameType
from omicron import tf

logger = logging.getLogger(__name__)

# todo: 将代码融合到omicron 2.1中


def init_jq():
    jq.auth(os.environ["JQ_ACCOUNT"], os.environ["JQ_PASSWORD"])


class Query:
    """
    ["code", "display_name", "name", "ipo", "end", "type"]
    """

    def __init__(self, df: pd.DataFrame, now: datetime.date = None):
        now = now or datetime.date.today()
        if df is None:
            self.recs = None
        else:
            self.recs = df[df.ipo < now]

    @property
    def codes(self):
        if self.recs is not None:
            return self.recs.index.tolist()
        else:
            return []

    def types(self, types: List[str]) -> "Query":
        """按类型过滤

        Args:
            types : "stock", "index", "etf", "lof", "fja", "fjb", "futures"
        """
        r = self.recs

        if r is None:
            return self

        self.recs = r[r.type.isin(types)]

        return self

    def only_cyb(self) -> "Query":
        r = self.recs

        if r is None:
            return self

        self.recs = r[r.index.str.startswith("300")]
        return self

    def only_st(self) -> "Query":
        r = self.recs

        if r is None:
            return self

        self.recs = r[r.display_name.str.contains("ST")]
        return self

    def only_kcb(self) -> "Query":
        r = self.recs

        if r is None:
            return self

        self.recs = r[r.index.str.startswith("688")]
        return self

    def exclude_st(self) -> "Query":
        if self.recs is None:
            return self

        self.recs = self.recs[~self.recs.display_name.str.contains("ST")]

        return self

    def exclude_cyb(self) -> "Query":
        r = self.recs

        if r is None:
            return self

        self.recs = r[~r.index.str.startswith("300")]
        return self

    def exclude_kcb(self) -> "Query":
        r = self.recs

        if r is None:
            return self

        self.recs = r[~r.index.str.startswith("688")]
        return self

    def exclude_exit(self, now: datetime.date) -> "Query":
        r = self.recs

        if r is None:
            return self

        self.recs = r[~(r.end <= now)]
        return self

    def name_like(self, name: str):
        r = self.recs

        if r is None:
            return self
        self.recs = r[r.display_name.str.contains(name)]

        return self


class Securities:
    _obj = None

    def __init__(self, path: str = None):
        self.store_path = path or "/data/securities.zarr"
        self.store = self.load()
        self._synced = [arrow.get(x).date() for x in self.store.attrs.get("synced", [])]

    @classmethod
    def get_instance(cls):
        if cls._obj is None:
            cls._obj = Securities()

        return cls._obj

    def sync(self, start: str, end: str):
        start = arrow.get(start).date()
        end = arrow.get(end).date()

        days = set(
            [
                tf.int2date(x).strftime("%Y-%m-%d")
                for x in tf.get_frames(start, end, FrameType.DAY)
            ]
        )

        # synced days which represented in str type
        synced = sorted(self.store.attrs.get("synced", []))

        if len(synced) > 0:
            logger.info("the store contains recs from %s to %s", synced[0], synced[-1])

        days = days - set(synced)

        if len(days) == 0:
            logger.info("nothing to update")
            return
        else:
            logger.info("updating %d days", len(days))

        init_jq()

        try:
            for day in days:
                logger.info("sync day: %s", day)
                df = jq.get_all_securities(
                    ["stock", "index", "etf", "lof", "fja", "fjb"], date=day
                )

                dtype = [
                    ("code", "U11"),
                    ("display_name", "U32"),
                    ("name", "U4"),
                    ("ipo", "<i8"),
                    ("end", "<i8"),
                    ("type", "U8"),
                ]

                secs = np.empty((len(df),), dtype=dtype)
                secs["code"] = df.index.values
                secs["display_name"] = df.display_name.values
                secs["name"] = df.name.values
                secs["ipo"] = [tf.date2int(x.to_pydatetime()) for x in df.start_date]
                secs["end"] = [tf.date2int(x.to_pydatetime()) for x in df.end_date]
                secs["type"] = df.type.values

                self.store[day] = secs
                synced.append(day)
                self._synced.append(arrow.get(day).date())
        except Exception as e:
            logger.exception(e)
            logger.warning("failed sync for %s", day)
            logger.warning("the dataframe is:%s", df)
        finally:
            self.store.attrs["synced"] = synced

            jq.logout()

    def load(self):
        try:
            return zarr.open(self.store_path, mode="a")
        except Exception:
            logger.warning("faield to open %s, store is re-created.", self.store_path)
            return None

    def close(self):
        self.store.close()

    def query(self, date: datetime.date) -> Query:
        """构建查询对象

        Args:
            date : 日期

        Returns:
            Query: 查询对象
        """
        if len(self._synced) == 0:
            return Query(None)

        synced = np.array(sorted(self._synced))
        last_day = synced[synced <= date]
        if len(last_day) == 0:
            return Query(None)

        key = last_day[-1].strftime("%Y-%m-%d")

        secs = self.store[key]

        df = pd.DataFrame([], secs.dtype.names)

        df.index = secs["code"]
        df["display_name"] = secs["display_name"]
        df["name"] = secs["name"]
        df["type"] = secs["type"]

        df["ipo"] = [tf.int2date(x) for x in secs["ipo"]]
        df["end"] = [tf.int2date(x) for x in secs["end"]]

        return Query(df, date)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename="/var/log/alpha/alpha.log")

    tf.service_degrade()
    # secs.query(datetime.date(2022, 3, 15)).types(["stock"]).name_like("银行").exclude_exit(datetime.date(2020, 3, 1)).codes

    # secs.sync("2022-03-01", "2022-03-10")
    fire.Fire({"sync": all_secs.sync})
