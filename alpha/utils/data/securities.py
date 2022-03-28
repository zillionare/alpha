import datetime
import os
import pickle

import arrow
import pandas as pd
import fire
import jqdatasdk as jq
from coretypes import FrameType
from omicron import tf
import logging
from typing import List

logger = logging.getLogger(__name__)

# todo: 将代码融合到omicron 2.1中

def init_jq():
    jq.auth(os.environ["JQ_ACCOUNT"], os.environ["JQ_PASSWORD"])

class Query:
    def __init__(self, df: pd.DataFrame):
        self.recs = df

    @property
    def codes(self):
        if self.recs is not None:
            return self.recs.index.tolist()
        else:
            return []

    def types(self, types: List[str])->'Query':
        """按类型过滤

        Args:
            types : "stock", "index", "etf", "lof", "fja", "fjb", "futures"
        """
        r = self.recs

        self.recs = r[r.type.isin(types)]

        return self

    def only_cyb(self)->'Query':
        r = self.recs

        self.recs = r[r.index.str.startswith("300")]
        return self

    def only_st(self)->'Query':
        r = self.recs

        self.recs = r[r.display_name.str.contains("ST")]
        return self

    def only_kcb(self)->'Query':
        r = self.recs

        self.recs = r[r.index.str.startswith("688")]
        return self

    def exclude_st(self)->'Query':
        self.recs = self.recs[~self.recs.display_name.str.contains("ST")]

        return self

    def exclude_cyb(self)->'Query':
        r = self.recs

        self.recs = r[~r.index.str.startswith("300")]
        return self

    def exclude_kcb(self)->'Query':
        r = self.recs

        self.recs = r[~r.index.str.startswith("688")]
        return self

    def name_like(self, name: str):
        r = self.recs

        self.recs = r[r.display_name.str.contains(name)]

        return self

class Securities:
    def __init__(self, path: str=None):
        self.store_path = path or "/data/securities.pkl"
        self.store = self.load()

    def save_securities(self, start: str, end: str):
        init_jq()

        tf.service_degrade()

        start = arrow.get(start).date()
        end = arrow.get(end).date()
        
        days = set([tf.int2date(x) for x in tf.get_frames(start, end, FrameType.DAY)])

        if os.path.exists(self.store_path):
            with open(self.store_path, "rb") as f:
                d = pickle.load(f)
        else:
            d = {}

        days = days - set(d.keys())

        try:
            for day in days:
                logger.info("sync day: %s", day)
                secs = jq.get_all_securities(["stock", "index", "etf", "lof", "fja", "fjb", "futures"], date=day)

                d[day] = secs
        except Exception as e:
            logger.exception(e)
        finally:
            with open(self.store_path, "wb") as f:
                pickle.dump(d, f)

            jq.logout()


    def load(self):
        with open(self.store_path, "rb") as f:
            return pickle.load(f)
    
    def query(self, date: datetime.date)->Query:
        """构建查询对象

        Args:
            date : 日期

        Returns:
            Query: 查询对象
        """
        return Query(self.store.get(date))

secs = Securities()

if __name__ == "__main__":
    fire.Fire({
        "sync": secs.save_securities
    })
