import os
from typing import List

from h2o_wave import Component, Q, ui

from alpha.web.routing import on

tabs = {"research": "研究", "bt": "回测", "stockpool": "股票池"}

js_file = os.path.join(os.path.dirname(__file__), "../assets/js/util.js")
with open(js_file, "r") as f:
    js = f.read()


async def render_header(active: str, q: Q) -> ui.HeaderCard:
    q.page["header"] = ui.header_card(
        box="header",
        title="Alpha",
        subtitle="Let's conquer the world",
        image="https://images.jieyu.ai/images/202204/logo-red-small.png",
        secondary_items=[
            ui.tabs(
                name="tabs",
                value=active,
                link=True,
                items=[
                    ui.tab(name=f"#{key}", label=f"{value}")
                    for key, value in tabs.items()
                ],
            )
        ],
        items=[
            ui.persona(
                name="account", title="Aaron Yang", subtitle="Researcher", size="xs"
            )
        ],
    )


@on("account")
async def change_persona(q: Q):
    q.page["meta"].dialog = ui.dialog(
        title="Change Persona",
        name="change_persona",
        items=[],
        blocking=True,
        closable=True,
    )

    await q.page.save()


def inlinejs(
    ps: str, requires: List[str] = None, targets: List[str] = None
) -> Component:
    """生成inline js

    Args:
        ps : page ad-hoc script

    Returns:
        ui.inline_script
    """
    global js
    return ui.inline_script(
        content=js + "\n" + ps,
        requires=requires,
        targets=targets,
    )
