from typing import Union

import dash_bootstrap_components as dbc
from dash import dcc
from dash.development.base_component import Component
from .widgets import header

def with_header(main: Component) -> Component:
    """构建页面。

    组装一个带header的页面。该页面上有一个location控件。

    Args:
        content : _description_
    """
    return dbc.Container(
        [
            dbc.Row(dbc.Col(header)),
            dbc.Row(dbc.Col(main)),
        ],
        style={"height": "100vh"},
        fluid=True,
    )

def with_header_left_sidebar(sidebar: Component, main: Component, width=2) -> Component:
    """构建页面。

    组装一个带header和left sidebar的页面。

    Args:
        main : main content
        sidebar : sidebar content
        width : width of sidebar
    """
    return dbc.Container(
        [
            dbc.Row(dbc.Col(header)),
            dbc.Row(
                [
                    dbc.Col(sidebar, width=width, style={
                        "margin-right": "1rem"
                    }),
                    dbc.Col(main, width=12-width),
                ],
                style={"height": "100vh", "flex-wrap": "nowrap"},
            ),
        ],
        style={"height": "100vh"},
        fluid=True,
    )

def with_header_right_sidebar(main: Component, sidebar: Component, width=2) -> Component:
    """构建页面。

    组装一个带header和right sidebar的页面。

    Args:
        main : main content
        sidebar : sidebar content
        width : width of sidebar
    """
    return dbc.Container(
        [
            dbc.Row(dbc.Col(header)),
            dbc.Row(
                [
                    dbc.Col(main, width=12-width),
                    dbc.Col(sidebar, width=width),
                ],
                style={"height": "100vh", "flex-wrap": "nowrap"},
            ),
        ],
        style={"height": "100vh"},
        fluid=True,
    )
