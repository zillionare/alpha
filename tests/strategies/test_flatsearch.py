# import unittest

# import arrow
# import cfg4py
# import omicron
# from omicron.models.timeframe import TimeFrame as tf
# from coretypes import FrameType
# from omicron.models.stock import Stock

# from alpha.config import get_config_dir
# from alpha.strategies.flatsearch import FlatSearchStrategy


# class TestFlatSearch(unittest.IsolatedAsyncioTestCase):
#     def setUp(self) -> None:
#         cfg4py.init(get_config_dir())
#         return super().setUp()

#     async def asyncSetUp(self) -> None:
#         self.setUp()
#         await omicron.init()
#         return await super().asyncSetUp()

#     async def test_fss(self):
#         fss = FlatSearchStrategy("flatsearch")
#         fss.reset_space()

#         fss.build_sample_space("/data/stocks/ds_even_30m_300_5000.pkl")

#         end = arrow.get("2021-08-19 15:00:00")
#         start = tf.shift(end, -300, FrameType.MIN30)
#         sec = Security("000519.XSHE")

#         bars = await sec.load_bars(start, end, FrameType.MIN30)

#         fss.predict(bars)
