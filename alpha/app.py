import asyncio
import os

import cfg4py
import dash_bootstrap_components as dbc
import omicron
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from asgiref.sync import AsyncToSync
from dash import Dash

from alpha.config import get_config_dir
from alpha.jobs import start_background_tasks
from alpha.web import routing

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.MATERIA, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    assets_folder=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "web/assets"
    ),
)


def start(port: int = 8050, host="0.0.0.0"):
    cfg4py.init(get_config_dir())

    AsyncToSync(omicron.init)()

    routing.build_blueprints()
    app.layout = routing.layout()
    app.title = "Alpha策略分析师"

    #start_background_tasks()
    # dev_tools_hot_reload=True
    app.run_server(
        host, port, dev_tools_hot_reload=True, dev_tools_hot_reload_interval=5
    )


if __name__ == "__main__":
    # fire.Fire({
    #     "start": start
    # })
    start()
