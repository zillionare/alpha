import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc
from alpha.web.components.scaffold import render_with_scaffold
from alpha.web.auth import get_user

location = dcc.Location(id="homepage", pathname="/research", refresh=False)


def render_home_page():
    global location
    return render_with_scaffold(location)


# add callback for toggling the collapse on small screens
@callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# @callback(Output("accountMenu", "label"), [Input(location, "pathname")])
# def _user(pathname):
#     return get_user() or "未登录"
