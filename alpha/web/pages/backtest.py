from typing import List

from h2o_wave import Q, ui, Expando
import datetime

from alpha.strategies import get_all_strategies
from alpha.web.widgets import render_header

from ..routing import StopPropagation, handle_on, on
import cfg4py
from traderclient import TradeClient
import uuid
import httpx
import arrow


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


async def init(q: Q):
    cfg = cfg4py.get_instance()


async def start_backtest(
    strategy: str,
    start: datetime.date,
    end: datetime.date,
    captital=1_000_000,
    commission=1e-4,
    params: dict = None,
):
    pass


async def get_bt_history(strategy: str):
    return [
        ui.button(
            "config_new_backtest",
            label="启动回测",
            primary=True,
            icon="play",
            value=strategy,
        ),
        ui.button(
            "hist1", label=uuid.uuid4().hex[:8] + "(2019-01-01, 2020-01-10)", link=True
        ),
        ui.button(
            "hist2", label=uuid.uuid4().hex[:8] + "(2020-01-01, 2021-01-10)", link=True
        ),
        ui.button(
            "hist3", label=uuid.uuid4().hex[:8] + "(2021-01-01, 2022-01-10)", link=True
        ),
    ]


def capitalize_first_letter(string):
    return string[0].upper() + string[1:]


async def render_left_panel(q: Q):
    all_strategies = get_all_strategies()
    strategy_items = []
    for name, desc, *_ in all_strategies:
        strategy_items.append(
            ui.expander(
                name,
                label=capitalize_first_letter(name),
                items=[
                    ui.text_xs(desc),
                    ui.separator(),
                    *(await get_bt_history(name)),
                ],
            )
        )

    q.page["left"] = ui.form_card("left", items=strategy_items)


async def content(q: Q):
    pass


@on("#bt")
async def backtest_view(q: Q):
    if q.client.layout != "bt":
        await init(q)
        set_layout(q)

    await render_header("bt", q)
    await render_left_panel(q)

    await q.page.save()


@on()
async def config_new_backtest(q: Q):
    q.page["meta"].dialog = ui.dialog(
        title="设置回测参数",
        name="backtest_setting",
        items=[
            ui.textbox("start_date", label="开始日期", value="2019-01-01"),
            ui.textbox("end_date", label="结束日期", value="2020-01-01"),
            ui.textbox("captital", label="本金", value="1,000,000"),
            ui.textbox("commission", label="手续费", value="0.015%"),
            ui.textbox(
                name="params",
                label="其它参数",
                multiline=True,
                tooltip="参数格式为json字符串，请传入对应的策略可解释的参数",
                placeholder='{"a": "b", "arr": [1, 2, 3], "dict": {"a": 1, "b": 2}}',
            ),
            ui.separator(label="", name="", width="100%", visible=True),
            ui.buttons(
                items=[
                    ui.button("start_backtest", label="启动回测", primary=True),
                    ui.button("cancel_backtest_setting", label="取消"),
                ],
                justify="end",
            ),
        ],
        blocking=True,
        closable=True,
    )
    await q.page.save()
    raise StopPropagation


@on()
async def cancel_backtest_setting(q: Q):
    q.page["meta"].dialog = None
    await q.page.save()
    raise StopPropagation


@on()
async def start_backtest(q: Q):
    start = arrow.get(q.args.start_date).date()
    end = arrow.get(q.args.end_date).date()
    capital = float(q.args.captital.replace(",", ""))
    strategy = q.args.strategy

    try:
        commission = float(q.args.commission)
    except ValueError:
        s = q.args.commission.strip()
        if s.endswith("%"):
            commission = float(s.strip("%")) / 100

    params = q.args.params

    q.page["meta"].dialog = None
    await q.page.save()
    raise StopPropagation
