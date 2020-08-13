"""Main module."""
import cfg4py
import fire
import omicron
from alpha.config import get_config_dir
from pyemit import emit
from sanic import Sanic
import logging

from alpha.plots.crossyear import cy

cfg = cfg4py.get_instance()
app = Sanic("alpha")

logger = logging.getLogger(__name__)
class Application(object):
    async def init(self, *args):
        logger.info("init alpha...")
        await omicron.init()
        await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn, start_server=False)
        await cy.start()

def start():
    cfg4py.init(get_config_dir())
    myapp = Application()
    app.register_listener(myapp.init, 'before_server_start')
    app.run(host='0.0.0.0', port=cfg.alpha.server.port,
            workers=cfg.alpha.server.workers)

if __name__ == "__main__":
    fire.Fire({
        'start': start
    })