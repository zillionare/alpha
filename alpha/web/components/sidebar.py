import dash_bootstrap_components as dbc
from dash import html

sidebar = html.Div(
    [
        dbc.Nav(
            [
                dbc.NavLink("研究", href="/research", active="exact"),
                dbc.NavLink("回测", href="/backtest", active="exact"),
                dbc.NavLink("票池", href="/pool", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style={"padding-top": "2rem"},
)
