"""Main module."""
import logging

import cfg4py
import fire
import omicron
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pyemit import emit
from sanic import Sanic, response

from alpha.config import get_config_dir
from alpha.core.monitors import mm
from alpha.plots import start_plot_scan
import alpha.web as handlers

cfg = cfg4py.get_instance()
app = Sanic("alpha")
app.config.RESPONSE_TIMEOUT = 300

logger = logging.getLogger(__name__)


class Application(object):
    def __init__(self):
        self.scheduler = None

    async def init(self, app, loop):
        logger.info("init alpha...")
        self.scheduler = AsyncIOScheduler({'event_loop': loop},
                                          timezone='Asia/Shanghai')
        self.scheduler.start()
        await omicron.init()
        await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn, start_server=False)

        mm.init(self.scheduler)
        start_plot_scan(self.scheduler)

        app.add_route(handlers.plot_command_handler, '/plot/<cmd>', methods=['POST'])
        app.add_route(handlers.add_monitor, '/monitor/add', methods=['POST'])
        app.add_route(handlers.remove_monitor, '/monitor/remove', methods=['POST'])
        app.add_route(handlers.list_monitors, '/monitor/list', methods=['GET'])
        app.add_route(self.jobs, '/jobs/<cmd>', methods=['POST'])
        app.add_route(handlers.get_stock_pool, '/stock_pool', methods=['GET'])
        app.add_route(handlers.fuzzy_match, '/common/fuzzy-match',
                      methods=['GET'])

    async def jobs(self, request, cmd):
        if cmd == 'list':
            result = self.list_jobs()
            return response.json(result, status=200)

    def list_jobs(self):
        result = []
        for job in self.scheduler.get_jobs():
            result.append([job.name, str(job.trigger), str(job.next_run_time)])

        return result

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
