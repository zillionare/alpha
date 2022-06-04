import logging

import cfg4py
import omicron
from h2o_wave import Expando, Q, app, main
from pyemit import emit

from alpha.config import get_config_dir
from alpha.core.executor import create_process_pool
from alpha.core.redislog import RedisLogReceiver
from alpha.web.pages import *
from alpha.web.pages.research import research_view
from alpha.web.routing import handle_on

logger = logging.getLogger(__name__)


async def on_client_connected(q: Q) -> None:
    q.user.theme = q.user.theme or "ember"

    # If no active hash present, render research.
    if q.args["#"] is None:
        await research_view(q)


async def on_startup():
    cfg = cfg4py.init(get_config_dir())
    await omicron.init()
    await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn)
    await RedisLogReceiver.listen()
    await create_process_pool(1)


async def on_shutdown():
    await omicron.stop()


@app("/", on_startup=on_startup, on_shutdown=on_shutdown)
async def serve(q: Q):
    if not q.client.initialized:
        await on_client_connected(q)
        q.client.initialized = True

    await handle_on(q)
