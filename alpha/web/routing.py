"""
routing module mimic the routing for the dash web servers.


"""

import glob
import importlib
import logging
import os

import dash_bootstrap_components as dbc
from dash import callback, dcc, html
from dash.dependencies import Input, Output

from alpha.web.homepage.view import render_home_page

logger = logging.getLogger(__name__)

# registry for all routes
routes = {}


def layout():
    router = dcc.Location(id="router", refresh=False)

    full_hw_style = {
        "height": "100vh",
        "width": "100vw",
    }

    return html.Div(
        [
            router,
            html.Div(id="page-content", style=full_hw_style),
        ],
        id="rootElement",
        style=full_hw_style,
    )


def on(pathname: str):
    """
    dispatch function.
    """

    def decorator(func):
        global routes

        routes[pathname] = func
        return func

    return decorator


@callback(
    Output("page-content", "children"),
    [Input("router", "pathname")],
)
def _routing(pathname: str):
    # ensure auth
    # from alpha.web import auth
    # if not auth.get_current_user():
    #     return auth.view.render()
    if pathname == "/":
        pathname = "/research"

    handler = routes.get(pathname, None)
    if handler is None:
        return render_home_page()

    return handler()


def build_blueprints():
    """
    collect all routes by import controller from web/*/controller.py
    """
    _dir = os.path.dirname(os.path.abspath(__file__))
    package_prefix = "alpha.web."

    for pattern in [f"{_dir}/**/controller.py", f"{_dir}/**/view.py"]:
        for pyfile in glob.glob(pattern):
            sub = pyfile.replace(f"{_dir}/", "").replace(".py", "").replace("/", ".")
            module_name = package_prefix + sub
            importlib.import_module(module_name)

    logger.info("blueprints loaded: %s", routes)
