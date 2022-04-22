import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc

from alpha.web.pages.layout import with_header


def render_home_page(main):
    return with_header(main)


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
