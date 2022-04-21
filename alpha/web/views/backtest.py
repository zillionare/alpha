from alpha.web import routing
from alpha.web.views.layout import with_header
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html


main = dbc.Container("backtest")

@routing.dispatch("/backtest")
def init_page():
    return with_header(main)
