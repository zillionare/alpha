"""Main module."""
import asyncio

import cfg4py
import fire
import omicron
from alpha.plots.crossyear import CrossYear
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from alpha.config import get_config_dir
from pyemit import emit
from sanic import Sanic, response
import logging

from alpha.core.monitor import monitor

cfg = cfg4py.get_instance()
app = Sanic("alpha")
app.config.RESPONSE_TIMEOUT = 300

logger = logging.getLogger(__name__)
class Application(object):
    async def init(self, app, loop):
        logger.info("init alpha...")
        scheduler = AsyncIOScheduler({'event_loop':loop}, timezone='Asia/Shanghai')
        scheduler.start()
        await omicron.init()
        await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn, start_server=False)

        monitor.init(scheduler)

        app.add_route(self.plot_command_handler, '/plot', methods=['POST'])
        app.add_route(self.monitor_add_watch,'/monitor/add', methods=['POST'])
        app.add_route(self.monitor_remove_watch,'/monitor/remove', methods=['POST'])
        app.add_route(self.monitor_list_watch,'/monitor/list', methods=['POST'])


    async def monitor_add_watch(self, request):
        """
        supported cmd:
            add, remove,list
        Args:
            request:

        Returns:

        """
        params = request.json
        code = params.get("code")
        plot = params.get("plot")
        frame_type = params.get("frame_type")
        flag = params.get("flag")

        if not all((code, plot, frame_type, flag)):
            return response.json("code, plot, frame_type, flag are required",
                                 status=401)
        try:
            await monitor.watch(**params)
            return response.json('done', status=200)
        except Exception as e:
            return response.json(e, status=500)

    async def monitor_remove_watch(self, request):
        params = request.json

        code = params.get("code")
        plot = params.get("plot")
        frame_type = params.get("frame_type")
        flag = params.get("flag")

        if not all((code, plot, frame_type, flag)):
            return response.json("code, plot, frame_type, flag are required",
                                 status=401)

        try:
            await monitor.remove(plot, code, frame_type, flag)
            return response.json('done', status=200)
        except Exception as e:
            return response.json(e, status=500)

    async def monitor_list_watch(self, request):
        params = request.json

        try:
            result = await monitor.list_watch(**params)
            return response.json(result, status=200)
        except Exception as e:
            return response.json(e, status=500)


    async def plot_command_handler(self, request):
        cmd = request.json.get("cmd")
        plot = request.json.get("plot")
        params = request.json
        del params['cmd']
        del params['plot']

        try:
            if plot == 'crossyear':
                cy = CrossYear()
                func = getattr(cy, cmd)
                if asyncio.iscoroutinefunction(func):
                    result = await func(params)
                else:
                    result = func(params)

                return response.json(body=result,status=200)
        except Exception as e:
            logger.exception(e)
            return response.json(e, status=500)
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