from h2o_wave import ui, Q

tabs = {"home": "研究", "bt": "回测", "stockpool": "股票池"}


async def header(active: str, q: Q) -> ui.HeaderCard:
    return ui.header_card(
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
                title="John Doe",
                subtitle="Developer",
                size="xs",
                image="https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&h=750&w=1260",
            )
        ],
    )
