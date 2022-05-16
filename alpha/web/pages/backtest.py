from typing import List

from h2o_wave import Q, ui
from sqlalchemy import all_

from alpha.strategies import get_all_strategies
from alpha.web.widgets import render_header

from ..routing import StopPropagation, handle_on, on


def set_layout(q: Q):
    q.page["meta"] = ui.meta_card(
        box="",
        title="Alpha",
        theme=q.user.theme,
        layouts=[
            ui.layout(
                breakpoint="xs",
                zones=[
                    ui.zone("header"),
                    ui.zone(
                        "body",
                        direction=ui.ZoneDirection.ROW,
                        size="100vh",
                        zones=[
                            ui.zone("left", size="20%"),
                            ui.zone("content"),
                        ],
                    ),
                ],
                height="100vh",
            )
        ],
    )
    q.client.layout = "#bt"


def init(q: Q):
    pass

async def get_bt_history(strategy: str, user: str):
    return []

def capitalize_first_letter(string):
    return string[0].upper() + string[1:]

async def render_left_panel(q: Q):
    all_strategies = get_all_strategies()
    strategy_items = []
    for name, desc, *_ in all_strategies:
        strategy_items.append(ui.expander(name, label=capitalize_first_letter(name), items=[
            ui.text_xs(desc),
            ui.separator(),
            *(await get_bt_history(name, None)),
        ])
        )

    q.page["left"] = ui.form_card("left", items=strategy_items)


async def content(q: Q):
    pass


@on("#bt")
async def backtest_view(q: Q):
    if q.client.layout != "bt":
        init(q)
        set_layout(q)

    await render_header("bt", q)
    await render_left_panel(q)

    await q.page.save()
