import logging
import unittest

import arrow
import cfg4py
import omicron
import pandas as pd
from backtesting.backtesting import Backtest
from omicron.core.types import FrameType
from omicron.models.security import Security

from alpha.config import get_config_dir
from alpha.strategies.z01 import Z01Strategy

cfg = cfg4py.get_instance()
logging.basicConfig(level=logging.INFO)


def to_dataframe(bars):
    df = pd.DataFrame(bars)
    df.set_index("frame", inplace=True)
    df.columns = ["Open", "High", "Low", "Close", "Volume", "Amount", "Factor"]
    return df.dropna()


class TestZ01(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        cfg4py.init(get_config_dir())
        await omicron.init()
        return await super().asyncSetUp()

    async def test_run(self):
        start = arrow.get("2019-1-3 15:00", tzinfo="Asia/Shanghai").datetime
        stop = arrow.now(tz="Asia/Shanghai").datetime

        bars = await Security("000001.XSHE").load_bars(start, stop, FrameType.MIN30)
        df = to_dataframe(bars)
        bt = Backtest(df, Z01Strategy)
        output = bt.run()
        print(output)

        print("\n")
        print(output["_trades"])

    async def test_optimize(self):
        start = arrow.get("2019-1-3 15:00", tzinfo="Asia/Shanghai").datetime
        stop = arrow.now(tz="Asia/Shanghai").datetime

        bars = await Security("000001.XSHE").load_bars(start, stop, FrameType.MIN30)
        df = to_dataframe(bars)
        bt = Backtest(df, Z01Strategy)

        stats, heatmap = bt.optimize(
            win=range(3, 10, 1),
            f1=[i / 5 for i in range(1, 10)],
            f2=[i / 5 for i in range(1, 10)],
            maximize="Equity Final [$]",
            max_tries=300,
            return_heatmap=True,
        )

        print(heatmap.sort_values().iloc[-3:])
