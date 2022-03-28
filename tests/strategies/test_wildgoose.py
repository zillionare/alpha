import unittest

from alpha.strategies.wildgoose import (
    AssetType,
    InvestStyle,
    OperateMode,
    WildGooseStrategy,
)


class WildGoose(unittest.IsolatedAsyncioTestCase):
    async def test_choose(self):
        s = WildGooseStrategy()
        codes = s.choose(
            modes=[OperateMode.CF, OperateMode.OF, OperateMode.LOF, OperateMode.ETF],
            styles=[
                InvestStyle.STOCK_FIRST,
                InvestStyle.STOCK_INDEX,
                InvestStyle.STOCK_ONLY,
            ],
        )

        print(len(codes))
        for code in codes:
            print(s.code2name(code))
