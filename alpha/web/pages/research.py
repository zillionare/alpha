import arrow
from coretypes import Frame, FrameType
from h2o_wave import Expando, Q, handle_on, on, ui
from omicron.models.stock import Stock
from plotly import graph_objects as go
from plotly import io as pio

from alpha.plotting.candlestick import Candlestick
from alpha.web.layout import add_card, clear_cards
from alpha.web.widgets import header


def left_panel():
    return ui.form_card(
        "left",
        items=[
            ui.inline(
                items=[
                    ui.combobox(
                        name="choose_symbol",
                        choices=["cyan", "mega", "yellow"],
                        width="80%",
                    ),
                    ui.button("add_favorite", icon="circleplus"),
                ],
                justify="between",
                inset=True,
            ),
            ui.table(
                name="favorites",
                columns=[
                    ui.table_column(name="name", label="Name"),
                    ui.table_column(name="symbol", label="Symbol"),
                ],
                rows=[
                    ui.table_row(name="row1", cells=["上证综指", "000001"]),
                    ui.table_row(name="row2", cells=["深证成指", "399001"]),
                    ui.table_row(name="row3", cells=["创业板指", "399006"]),
                ],
            ),
        ],
    )


def right_panel():
    # Render the plot as an HTML.
    return ui.MarkdownCard("right", title="诊股", content="right panel")


async def content(q: Q):
    code = q.args["choose_symbol"] or q.client.research.code
    n = q.client.research.nbars
    frame_type = FrameType(q.client.research.frame_type)
    end = q.client.research.end

    bars = await Stock.get_bars(code, n, frame_type, end, unclosed=True)
    cs = Candlestick(
        bars, title=f"{Stock(code).display_name} {frame_type.value.upper()}"
    )

    # config = {"scrollZoom": False, "showLink": False, "displayModeBar": False}
    html = pio.to_html(cs.figure, validate=False, include_plotlyjs="cdn")
    graph = ui.frame(content=html, height="90vh", width="100%")
    return ui.form_card("content", items=[toolbar(), graph])


def toolbar():
    return ui.inline(
        items=[
            ui.text("起始日期"),
            ui.textbox("start", mask="99 年 99 月 99 日", width="10vw"),
            ui.text("结束日期"),
            ui.textbox("end", mask="99 年 99 月 99 日", width="10vw"),
            ui.button("fastbackward", caption="后退30", icon="Rewind"),
            ui.button("backward", caption="后退一", icon="PlayReverse", primary=False),
            ui.button("play", caption="前进一", icon="Play", primary=False),
            ui.button("fastforward", caption="前进30", icon="FastForward", primary=False),
            ui.button("frame_type", caption="周期", icon="CalendarSettings", primary=False),
            ui.button("settings", caption="设置", icon="Settings", primary=False),
        ],
        justify="end",
        inset=True
    )


def set_layout(q: Q):
    q.page["meta"] = ui.meta_card(
        box="",
        title="Alpha",
        theme=q.user.theme,
        scripts=[ui.script("https://cdn.plot.ly/plotly-2.11.1.min.js")],
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
                            ui.zone("content", size="60%"),
                            ui.zone("right", size="20%"),
                        ],
                    ),
                ],
            )
        ],
    )


@on("#research")
async def research_view(q: Q):
    if not q.client.initialized:
        q.client.initialized = True
        q.client.research = Expando(
            {
                "nbars": 120,
                "frame_type": "1d",
                "end": arrow.now().date(),
                "code": "000001.XSHG",
            }
        )

    q.page.drop()

    set_layout(q)
    q.page["header"] = header("research")
    q.page["left"] = left_panel()
    q.page["right"] = right_panel()
    q.page["content"] = await content(q)

    await q.page.save()
