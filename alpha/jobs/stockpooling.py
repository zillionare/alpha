import datetime

from coretypes import Frame, FrameType
from omicron import tf
from omicron.models.stock import Stock


async def ting():
    """10日内涨停个股"""
    print(f"hello world at {datetime.datetime.now()}")
