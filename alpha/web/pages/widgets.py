from typing import Any, List

import dash_bootstrap_components as dbc
from dash import html
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, html

from alpha.web.models.session import get_user


def make_form(
    idprefix: str,
    labels: List[str],
    types: List[Any] = None,
    defaults: List[str] = None,
    tooltips: List[str] = None,
    icons: List[str] = None,
):
    """转换成一个container包含的多行输入框。

    icons使用font-awesome的图标,只传入图标名称，无须前缀(fas fa-)
    Args:
    """
    rows = []

    if types is None:
        types = ["text"] * len(labels)

    if tooltips is None:
        tooltips = [""] * len(labels)

    if icons is None:
        icons = [""] * len(labels)

    if defaults is None:
        defaults = [""] * len(labels)

    assert len(labels) == len(types) == len(tooltips) == len(icons) == len(defaults)

    label_len = max([len(x) for x in labels])

    for i, (label, _type, tooltip, icon, default) in enumerate(
        zip(labels, types, tooltips, icons, defaults)
    ):
        assert _type in [
            "text",
            "number",
            "password",
            "email",
            "tel",
            "url",
            "search",
            "checkbox",
        ], "不支持的控件类型: {}".format(_type)

        if _type == "checkbox":
            row = dbc.InputGroup(
                [
                    dbc.Checkbox(id=f"{idprefix}-{i}", label=f"{label}", value=default),
                    dbc.Tooltip(tooltip, target=f"{idprefix}-{i}"),
                ],
                style={"padding": "0.5rem 0", "align-items": "center"},
            )
        else:
            row = dbc.InputGroup(
                [
                    html.I(
                        className="fas fa-{}".format(icon),
                        style={"margin-right": "0.5em"},
                    ),
                    dbc.InputGroupText(
                        label,
                        style={"width": f"{label_len}rem", "margin-right": "0.5em"},
                    ),
                    dbc.Input(id=f"{idprefix}-{i}", type=_type, placeholder=default),
                    dbc.Tooltip(tooltip, target=f"{idprefix}-{i}"),
                ],
                style={"width": "85vh", "align-items": "center"},
            )

        rows.append(row)

    return dbc.Container(rows, fluid=True, class_name="small")


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

backtest_menu = dbc.DropdownMenu(
    [
        dbc.DropdownMenuItem(
            "网格交易",
            href="/backtest#gridtrade",
        )
    ],
    nav=True,
    in_navbar=True,
    label="回测",
    id="backtest-menu",
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
                    [
                        backtest_menu,
                        dbc.NavItem(dbc.NavLink("股票池", href="/pool")),
                        dbc.NavItem(dbc.NavLink("研究", href="/research", active=True)),
                        account_menu,
                    ],
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
