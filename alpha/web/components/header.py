import dash_bootstrap_components as dbc
from dash import html

# account/logout dropdown menu in the navbar
account_menu = dbc.DropdownMenu(
    [
        dbc.DropdownMenuItem(
            "Account",
            header=True,
            disabled=True,
            style={"font-weight": "bold"},
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
