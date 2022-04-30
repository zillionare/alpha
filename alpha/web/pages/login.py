import dash
import dash_bootstrap_components as dbc
from dash import callback, dcc, html
from dash.dependencies import Input, Output, State
from alpha.web import routing

from ..models.session import get_user, login_user, remove_session

form = dbc.Card(
    [
        dbc.CardHeader(html.H3("Welcome to Alpha!")),
        dbc.CardBody(
            [
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("用户："),
                        dbc.Input(
                            placeholder="Username", type="text", id="usernameBox"
                        ),
                    ],
                    class_name="sm-3",
                    style={"padding-top": "1rem", "padding-bottom": "1rem"},
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("密码："),
                        dbc.Input(type="password", id="passwordBox"),
                    ],
                    class_name="sm-3",
                    style={"padding-top": "1rem", "padding-bottom": "1rem"},
                ),
            ]
        ),
        dbc.CardFooter(
            dbc.Button(" 登  录 ", id="loginButton", size="sm"),
            style={"textAlign": "right"},
        ),
    ]
)


def render():
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

    return layout


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
    if (
        (n_clicks and n_clicks > 0)
        or (usernameSubmit and usernameSubmit > 0)
        or (passwordSubmit and passwordSubmit > 0)
    ):
        if get_user() is None:
            response = dash.callback_context.response
            if login_user(response, username, password):
                return dash.no_update, dash.no_update

        return True, True
    else:
        return dash.no_update, dash.no_update


@routing.dispatch("/logout")
def logout():
    """
    logout function.
    """
    if get_user():
        remove_session()

    return render()


@routing.dispatch("/login")
def login():
    """
    login function.
    """
    return render()
