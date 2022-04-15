"""
routing module mimic the routing for the dash web servers.


"""

import glob
import importlib
import os

import dash_bootstrap_components as dbc
from dash import callback, dcc, html
from dash.dependencies import Input, Output

# registry for all routes
routes = {}
router = dcc.Location(id="router", refresh=False)

full_hw_style = {
    "height": "100vh",
    "width": "100vw",
}
layout = html.Div(
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


@callback(Output("page-content", "children"), [Input(router, "pathname")])
def _routing(pathname):
    from alpha.web import auth, homepage

    print(f"in _routing: pathname: {pathname}")
    # ensure auth
    if not auth.get_current_user():
        print("returning auth layout")
        return auth.layout

    handler = routes.get(pathname, None)
    if handler is None:
        return homepage.layout

    return handler()


def build_blueprints():
    """
    collect all routes by import controller from web/*/controller.py
    """
    _dir = os.path.dirname(os.path.abspath(__file__))
    package_prefix = "alpha.web."
    for pyfile in glob.glob(f"{_dir}/**/controller.py"):
        sub = pyfile.replace(f"{_dir}/", "").replace(".py", "").replace("/", ".")
        module_name = package_prefix + sub
        importlib.import_module(module_name)
