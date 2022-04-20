import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

from alpha.web.auth.models import get_user

# account/logout dropdown menu in the navbar
account_menu = dbc.DropdownMenu(
    [
        dbc.DropdownMenuItem(
            "Account",
            header=True,
            disabled=True,
            style={"font-weight": "bold"},
            id="account-menu-header",
        ),
        dbc.DropdownMenuItem("Logout", href="/logout"),
    ],
    nav=True,
    in_navbar=True,
    label="Account",
    id="accountMenu",
)


header = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="assets/img/logo.png", height="32px")),
                        dbc.Col(dbc.NavbarBrand("Alpha", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler2", n_clicks=0),
            dbc.Collapse(
                dbc.Nav(
                    [dbc.NavItem(dbc.NavLink("Home", href="/")), account_menu],
                    className="ms-auto",
                    navbar=True,
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ],
    ),
    color="primary",
    dark=True,
    # bootstrap navbar contains 2rem padding, which cause navbar too large
    style={"padding": 0, "margin-bottom": "20px"},
)


@callback(
    Output("account-menu-header", "children"),
    Output("accountMenu", "label"),
    Input("router", "pathname"),
)
def update_user(nclicks):
    user = get_user() or "未登录"
    return user, user
