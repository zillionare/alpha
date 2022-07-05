""" to start alpha itself as a server, not a module
"""
from h2o_wave import Expando, Q, app, main
from alpha.web.pages.research import research_view
import cfg4py
import omicron
from pyemit import emit
from alpha.core.redislog import RedisLogReceiver
from alpha.app import start



async def on_connected(q: Q) -> None:
    q.user.theme = q.user.theme or "ember"

    # If no active hash present, render research.
    if q.args["#"] is None:
        await research_view(q)


async def on_startup():
    from alpha.config import get_config_dir

    cfg4py.init(get_config_dir())
    cfg = cfg4py.get_instance()
    await omicron.init()
    await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn)
    await RedisLogReceiver.listen()

async def on_shutdown():
    await omicron.close()


start(
    "/",
    on_app_startup=on_startup,
    on_app_shutdown=on_shutdown,
    on_client_connected=on_connected,
)
