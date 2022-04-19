from .header import header
from .sidebar import sidebar
import dash_bootstrap_components as dbc
from dash import dcc
from typing import Union
from dash.development.base_component import Component


def render_with_scaffold(
    location: dcc.Location, content: Union[str, Component] = ""
) -> Component:
    """构建页面。

    组装一个带header, sidebar和main_content的页面。该页面上有一个location控件。

    Args:
        location : _description_
        pathname : _description_
        content : _description_
    """
    return dbc.Container(
        [
            location,
            dbc.Row(dbc.Col(header)),
            dbc.Row(
                [
                    dbc.Col(sidebar, width=2),
                    dbc.Col(content, width=10),
                ]
            ),
        ],
        style={"height": "100vh"},
        fluid=True,
    )
