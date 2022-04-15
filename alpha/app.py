import os

import dash_bootstrap_components as dbc
from dash import Dash

from alpha.web import routing

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.MATERIA],
    suppress_callback_exceptions=True,
)


def start(port: int = 8050, host="0.0.0.0"):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    static_folder = os.path.join(cur_dir, "web/static")

    routing.build_blueprints()
    app.layout = routing.layout
    app.title = "Alpha策略分析师"

    # dev_tools_hot_reload=True
    app.run_server(
        host, port, dev_tools_hot_reload=True, dev_tools_hot_reload_interval=5
    )


if __name__ == "__main__":
    # fire.Fire({
    #     "start": start
    # })
    start()
