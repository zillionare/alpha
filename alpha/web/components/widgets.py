from typing import Any, List

import dash_bootstrap_components as dbc
from dash import html


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
