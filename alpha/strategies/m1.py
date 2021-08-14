import datetime
from alpha.core.features import fillna
from omicron.core.types import Frame, FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security
from omicron.core.timeframe import tf
from alpha.utils import round
import arrow
import numpy as np
import asyncio
import omicron
import cfg4py
from alpha.config import get_config_dir

cft = cfg4py.get_instance()

from pymilvus import Milvus
from pymilvus_orm import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    list_collections,
)

milvus = Milvus("172.17.0.1", "19530")


def init_stocks_collection():
    id_field = {
        "name": "id",
        "type": DataType.INT64,
        "auto_id": True,
        "is_primary": True,
    }

    code_field = {"name": "code", "type": DataType.INT32}

    start_field = {"name": "start", "type": DataType.INT32}

    feature_field = {
        "name": "features",
        "type": DataType.FLOAT_VECTOR,
        "metric_type": "L2",
        "params": {"dim": 60},
        "indexes": [{"metric_type": "L2"}],
    }

    fields = {"fields": [id_field, code_field, start_field, feature_field]}
    if milvus.has_collection("stock"):
        milvus.drop_collection("stock")

    milvus.create_collection("stock", fields=fields)


async def build(total: int) -> int:
    """找到涨停个股并插入milvus"""
    secs = Securities()
    stop = arrow.now().shift(days=-10)
    start = tf.day_shift(stop, -260)

    count = 0
    for code in secs.choose(
        ["stock"], exclude_688=True, exclude_st=True, exclude_exit=True
    ):
        sec = Security(code)

        bars = await sec.load_bars(start, stop, FrameType.DAY)

        buy_limit = np.where(np.isfinite(bars["close"]), bars["close"], 0)
        buy_limit = np.array([round(f * 1.1, 2) for f in buy_limit])
        pos = np.argwhere(
            (buy_limit[:-1] - bars["close"][1:] <= 0.009) & (buy_limit[:-1] != 0)
        )

        for i in pos.ravel():
            if i < 30:
                continue

            close = bars["close"][i - 30 : i]
            if np.count_nonzero(np.isfinite(close)) < len(close) * 0.9:
                continue

            vec = []
            close = fillna(close.copy())
            vec.extend((close / close[0]).tolist())

            volume = bars["volume"][i - 30 : i]
            vec.extend((volume / volume[0]).tolist())

            icode = int("1" + code.split(".")[0])
            idate = tf.date2int(bars[i - 29]["frame"])
            milvus.insert(
                "stock",
                [
                    {"name": "code", "type": DataType.INT32, "values": [icode]},
                    {"name": "start", "type": DataType.INT32, "values": [idate]},
                    {
                        "name": "features",
                        "type": DataType.FLOAT_VECTOR,
                        "values": [vec],
                    },
                ],
            )
            count += 1
            if count % 100 == 0:
                print(f"inserted {count} rows")
        if count >= total:
            break

    print("dataset ready")
    milvus.flush(["stock"])
    print("show status:", milvus.get_collection_stats("stock"))
    IVF_FLAT = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }
    milvus.create_index("stock", "features", IVF_FLAT)
    milvus.flush(["stock"])


async def find():
    milvus.load_collection("stock")

    vec = None
    # dsl = {
    #     "bool": {
    #         "must": [
    #             {
    #                 "vector": {
    #                     "features": {
    #                         "topK": 5,
    #                         "query": [vec],
    #                         "metric_type": "L2",
    #                         "params": {
    #                             "nprobe":10
    #                         }
    #                     }
    #                 }
    #             }
    #         ]
    #     }
    # }

    topK = 5
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    code = "000004.XSHE"
    sec = Security(code)
    start = datetime.date(2018, 1, 1)
    end = datetime.date(2021, 8, 1)

    bars = await sec.load_bars(start, end, FrameType.DAY)

    for i in range(30, len(bars)):
        close = bars["close"][i - 30 : i]
        if np.count_nonzero(np.isfinite(close)) < len(close) * 0.9:
            continue

        vec = []
        close = fillna(close.copy())
        vec.extend(close / close[0])

        volume = bars["volume"][i - 30 : i]
        vec.extend(volume / volume[0])

        res = milvus.search_with_expression(
            "stock",
            [vec],
            "features",
            param=search_params,
            limit=2,
            output_fields=["code", "start"],
        )
        # res = collection.search(vec, "features", search_params, topK)
        print(res)


async def main():
    cfg4py.init(get_config_dir())

    await omicron.init()

    init_stocks_collection()
    # await build(5)
    await find()


if __name__ == "__main__":
    asyncio.run(main())
