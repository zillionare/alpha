import os

import cfg4py
import omicron
from aioproc import expose

from alpha.config import get_config_dir


async def init():
    cfg4py.init(get_config_dir())
    await omicron.init()


async def scan(strategy: str, *args, **kwargs):
    from alpha.cli import create_strategy

    print(
        f"{strategy}({os.getpid()}) strategy started with args:{args}, kwargs:{kwargs}"
    )
    await init()
    s = create_strategy(strategy)
    await s.scan(*args, **kwargs)
    print(f"{strategy}({os.getpid()}) strategy finished")


expose({"scan": scan})
