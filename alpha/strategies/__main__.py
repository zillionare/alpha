import omicron
from aioproc import expose
import cfg4py
from alpha.config import get_config_dir
import os

async def init():
    cfg4py.init(get_config_dir())
    await omicron.init()

async def scan(strategy:str, *args):
    from alpha.cli import create_strategy

    print(f"{strategy} strategy started: {os.getpid()}")
    await init()
    s = create_strategy(strategy)
    await s.scan(*args)
    print(f"{strategy} strategy finished: {os.getpid()}")

expose({"scan": scan})
