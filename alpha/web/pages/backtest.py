import asyncio
from collections import defaultdict
import datetime
import logging
import random
import uuid
from typing import Dict, List, Tuple

import arrow
import cfg4py
import numpy as np
from coretypes import FrameType
from h2o_wave import Expando, Q, data, site, ui
from omicron import tf
from omicron.models.stock import Stock
from traderclient import TraderClient
from alpha.core.const import hs300

from alpha.core.commons import DataEvent
from alpha.strategies import (
    create_strategy_by_name,
    get_all_strategies,
    run_backtest_remote,
)
from alpha.web.widgets import in_app_notify, notify, render_header

from ..routing import StopPropagation, on

logger = logging.getLogger(__name__)
cfg = cfg4py.get_instance()


async def init(q: Q):
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
                            ui.zone(
                                "content",
                                zones=[
                                    ui.zone("metrics", size="20%"),
                                    ui.zone("chart", size="80%"),
                                ],
                            ),
                        ],
                    ),
                ],
                height="100vh",
            )
        ],
    )
    q.client.layout = "#bt"
    q.client.bt_progress = []


async def get_bt_history(q: Q) -> Dict:
    """获取回测历史记录

    按策略名分组获取回测历史记录。获取结果将缓存在`q.client.bt_history`中。
    Args:
        q: Query对象
    """
    url = cfg.backtest.url.rstrip("/")
    # url = "http://192.168.100.112:3180/backtest/api/trade/v0.3"
    accounts = TraderClient.list_accounts(url, cfg.backtest.admin_token)
    # infos = [
    #     TraderClient(url, item["account_name"], item["token"], is_backtest=False).info()
    #     for item in accounts
    # ]

    q.client.accounts = {}
    infos = []
    for item in accounts:
        try:
            infos.append(
                TraderClient(
                    url, item["account_name"], item["token"], is_backtest=False
                ).info()
            )
            q.client.accounts[item["account_name"]] = item["token"]
        except Exception as e:
            logger.exception(e)
    q.client.history = {item["name"]: item for item in infos if item.get("last_trade_date")}

    return q.client.history


def capitalize_first_letter(string):
    return string[0].upper() + string[1:]


async def render_left_panel(q: Q):
    all_strategies = get_all_strategies()

    history = await get_bt_history(q)

    strategy_items = []

    groups = defaultdict(list)
    # group history belongs same strategy into same expander
    for name, desc, *_ in all_strategies:
        for k, v in history.items():
            if k.startswith(name):
                # form the links
                groups[name].append(
                    ui.button(
                        "plot_history",
                        value=k,
                        label=f"{k} {v['start']}~{v['last_trade']}",
                        link=True,
                        tooltip=f"{v['principal']/10000}万({v['ppnl']:.2%})",
                    )
                )

        # generate expander groups for the strategy
    for name, *_ in all_strategies:
        strategy_items.append(
            ui.expander(
                name,
                label=capitalize_first_letter(name),
                items=[
                    ui.text_xs(desc),
                    ui.separator(),
                    ui.button(
                        "config_new_backtest",
                        label="启动新回测",
                        primary=True,
                        icon="play",
                        value=name,
                    ),
                    *groups.get(name, []),
                ],
            )
        )

    import_desc = "如果您的策略仅仅使用了zillionare-backtest来进行回测，而不是基于alpha来编写的，您可以在这里导入并查看结果。"
    strategy_items.append(
        ui.expander(
            "external_backtest",
            label="回测结果导入",
            items=[
                ui.text_xs(import_desc),
                ui.separator(),
                ui.button(
                    "import_backtest_result",
                    label="导入",
                    primary=True,
                    icon="play",
                    value=name,
                ),
            ],
        )
    )
    q.page["left"] = ui.form_card("left", items=strategy_items)


async def get_baseline_assets(
    baseline: str, start: datetime.date, end: datetime.date
) -> np.ndarray:
    """获取并计算在`start`到`end`区间的`baseline`的资产收益数据

    Args:
        baseline: baseline品种代码
        start: 回测超始时间
        end: 回测结束时间

    Returns:
        _description_
    """


async def render_chart(q: Q, account: str, token: str):
    """检索并绘制已完成的历史回测图

    Args:
        q: Query对象
        account: 回测账户名
        token: 回测账户token
    """
    baseline = q.user.backtest.baseline or hs300
    baseline_assets = await get_baseline_assets(baseline)
    q.page["charts"] = ui.plot_card(
        box="content",
        title="Line, groups",
        data=data("date series assets", -1000),
        plot=ui.plot(
            [
                ui.mark(
                    type="line",
                    x="=date",
                    y="=assets",
                    color="=series",
                    x_scale="time-category",
                )
            ]
        ),
    )

    start = datetime.date(2022, 1, 4)
    end = datetime.date(2022, 5, 10)
    principal = 1_000_000
    params = {"code": hs300, "frame_type": "1d"}
    token = uuid.uuid4().hex
    name = f"sma-v1-{token[-4:]}"
    event = await run_backtest_remote("sma", start, end, principal, params, name, token)

    n = tf.count_day_frames(start, end)
    bars = await Stock.get_bars(hs300, n, FrameType.DAY, end=end)
    baseline = np.array(
        [
            (date, "沪深300", round(nv.item(), 3))
            for date, nv in zip(bars["frame"], (bars["close"] / bars["open"][0]))
        ],
        dtype=[("frame", "O"), ("series", "O"), ("assets", "f8")],
    )

    while True:
        try:
            await event.wait()
            event.clear()
            msg = event.data
            logger.info("received msg:%s", msg)
            if msg["event"] == "finished":
                break
            elif msg["event"] == "progress":
                frame = msg["frame"]

                pos = np.argwhere(baseline["frame"] == frame).flatten()
                assert len(pos) > 0
                i = pos[0]

                q.page["content"].data[-1] = [
                    baseline[i][0].isoformat(),
                    baseline[i][1],
                    baseline[i][2].item(),
                ]
                q.page["content"].data[-1] = [
                    frame.isoformat(),
                    hs300,
                    round(msg["info"]["assets"] / principal, 3),
                ]

            await q.page.save()
        except Exception as e:
            logger.exception(e)


@on("#bt")
async def backtest_view(q: Q):
    """the landing page for 回测 tab"""
    if q.client.layout != "#bt":
        await init(q)

    await render_header("bt", q)
    await render_left_panel(q)

    hs300 = "399300.XSHE"
    bars = await Stock.get_bars(hs300, 120, FrameType.DAY)
    equities = [
        (date.isoformat(), "沪深300", round(nv.item(), 3))
        for date, nv in zip(bars["frame"], (bars["close"] / bars["open"][0]))
    ]
    # await render_chart(q)

    await q.page.save()


@on()
async def config_new_backtest(q: Q):
    q.client.strategy = q.args.config_new_backtest
    q.page["meta"].dialog = ui.dialog(
        title="设置回测参数",
        name="backtest_setting",
        items=[
            ui.textbox("start_date", label="开始日期", value="2019-01-01"),
            ui.textbox("end_date", label="结束日期", value="2020-01-01"),
            ui.textbox("principal", label="本金", value="1,000,000"),
            ui.textbox("commission", label="手续费", value="0.015%"),
            ui.textbox(
                name="params",
                label="其它参数",
                multiline=True,
                tooltip="参数格式为json字符串，请传入对应的策略可解释的参数",
                placeholder='{"a": "b", "arr": [1, 2, 3], "dict": {"a": 1, "b": 2}}',
            ),
            ui.textbox(
                name="notes",
                label="备注",
                tooltip="为回测设置备注信息，方便区分",
                multiline=True,
                placeholder="此次回测在v1的基础上，改变了learning_rate",
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
    principal = float(q.args.principal.replace(",", ""))
    strategy = q.client.strategy
    notes = q.args.notes

    try:
        commission = float(q.args.commission)
    except ValueError:
        s = q.args.commission.strip()
        if s.endswith("%"):
            commission = float(s.strip("%")) / 100

    params = q.args.params

    q.page["meta"].dialog = None
    s = create_strategy_by_name(strategy)
    if True:
        in_app_notify(q, "something is wrong", "error")

    await q.page.save()

    try:
        token = uuid.uuid4().hex
        name = f"{s.name}-{s.version}-{token[-4:]}"
        event = await run_backtest_remote(
            strategy, start, end, principal, params, name, token
        )

        while True:
            await event.wait()
            data = event.data
            if data["event"] == "finished":
                break
            else:
                await update_backtest_progress(q, data)

        q.user.backtest.history.append(
            {
                "name": name,
                "strategy": strategy,
                "start": start,
                "end": end,
                "principal": principal,
                "commission": commission,
                "params": params,
                "notes": notes,
                "token": token,
                "status": "running",
            }
        )
    except Exception as e:
        # since this might be a long run, we use notification to show the error
        notify(q, f"backtest {name} failed")
        logger.exception(e)

    await q.page.save()
    raise StopPropagation


async def update_backtest_progress(q: Q, data: dict):
    if data["event"] == "started":
        q.client.bt_progress = []
    elif data["event"] == "progress":
        # get the data from the server
        q.client.bt_progress.append(
            (len(q.client.bt_progress), "0", random.random() * 10)
        )
        q.client.bt_progress.append(
            (len(q.client.bt_progress), "1", random.random() * 10)
        )
    elif data["event"] == "trade":
        pass

@on()
async def plot_history(q: Q):
    account = q.args.plot_history

    info = q.client.bt_history.get(account)
    start = info["start"]
    end = info.get("last_trade_date")
    baseline = q.user.backtest.baseline or hs300

    bl_bars = await Stock.get_bars_in_range(baseline, start, end, FrameType.DAY)
    bl_equities = [(bar["frame"], "基准", bar["close"].item()/bl_bars[0]["open"]) for bar in bl_bars]

    token = q.client.accounts.get(account)
    client = TraderClient(cfg.backtest.url, info["name"], token, is_backtest=False)
    assets = client.bills()

    

