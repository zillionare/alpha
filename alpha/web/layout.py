from h2o_wave import ui
from typing import List, Optional


def meta(theme):
    return ui.meta_card(
        box="",
        title="Alpha",
        theme=theme,
        layouts=[
            ui.layout(
                breakpoint="xs",
                zones=[
                    ui.zone("header"),
                    ui.zone(
                        "body",
                        direction=ui.ZoneDirection.COLUMN,
                        zones=[ui.zone("sidebar"), ui.zone("content")],
                    ),
                    ui.zone("footer"),
                ],
            ),
            ui.layout(
                breakpoint="xl",
                width="1200px",
                zones=[
                    ui.zone("header"),
                    ui.zone(
                        "body",
                        direction=ui.ZoneDirection.ROW,
                        zones=[
                            ui.zone("sidebar", size="30%"),
                            ui.zone("content", size="70%"),
                        ],
                    ),
                ],
            ),
        ],
    )


# Use for page cards that should be removed when navigating away.
# For pages that should be always present on screen use q.page[key] = ...
def add_card(q, name, card) -> None:
    q.client.cards.add(name)
    q.page[name] = card


# Remove all the cards related to navigation.
def clear_cards(q, ignore: Optional[List[str]] = []) -> None:
    if not q.client.cards:
        return
    for name in q.client.cards.copy():
        if name not in ignore:
            del q.page[name]
            q.client.cards.remove(name)
