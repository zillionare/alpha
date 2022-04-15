from cProfile import label

import dash
import dash_bootstrap_components as dbc
from dash import callback, dcc, html
from dash.dependencies import Input, Output, State

from .models import get_current_user, login_user

form = dbc.Card(
    [
        dbc.CardHeader(html.H2("Alpha分析系统")),
        dbc.CardBody(
            [
                dbc.Label("用户名", size="sm"),
                dbc.Input(
                    id="usernameBox",
                    placeholder="用户名为手机号、邮箱",
                    type="text",
                    size="md",
                    className="mb-3",
                ),
                dbc.Label("密码", size="sm"),
                dbc.Input(
                    id="passwordBox",
                    placeholder="密码",
                    type="password",
                    size="md",
                    className="mb-3",
                ),
            ]
        ),
        dbc.CardFooter(
            dbc.Button(" 登  录 ", id="loginButton", size="sm"),
            style={"textAlign": "right"},
        ),
    ]
)
layout = dbc.Container(
    dbc.Row(
        [
            dcc.Location(id="urlLogin", pathname="/login", refresh=True),
            dbc.Row(
                [
                    dbc.Col(width=4),
                    dbc.Col(form, width=4, align="center"),
                    dbc.Col(width=4),
                ]
            ),
        ],
        style={"height": "100vh"},
    )
)


@callback(
    Output("urlLogin", "pathname"),
    Input("loginButton", "n_clicks"),
    [State("usernameBox", "value"), State("passwordBox", "value")],
    suppress_callback_exceptions=True,
)
def on_login(n_clicks, username, password):
    if n_clicks and n_clicks > 0:
        response = dash.callback_context.response
        if login_user(response, username, password):
            # jump to index page
            return "/"

    return dash.no_update


@callback(
    Output("usernameBox", "invalid"),
    Output("passwordBox", "invalid"),
    Input("loginButton", "n_clicks"),
    Input("usernameBox", "n_submit"),
    Input("passwordBox", "n_submit"),
    [State("usernameBox", "value"), State("passwordBox", "value")],
)
def update_output(n_clicks, usernameSubmit, passwordSubmit, username, password):
    if (n_clicks and n_clicks > 0) or (usernameSubmit and usernameSubmit > 0) or (passwordSubmit and passwordSubmit > 0):
        if get_current_user() is None:
            response = dash.callback_context.response
            if login_user(response, username, password):
                return dash.no_update, dash.no_update

        return True, True
    else:
        return dash.no_update, dash.no_update
