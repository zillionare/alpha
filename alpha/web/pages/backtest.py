import asyncio
import datetime
import itertools
import json
import logging
import random
import uuid
from collections import defaultdict
from typing import Dict, List, Tuple

import arrow
import cfg4py
import numpy as np
from coretypes import FrameType
from h2o_wave import Expando, Q, data, site, ui
from omicron import tf
from omicron.models.stock import Stock
from traderclient import TraderClient

from alpha.core.commons import DataEvent
from alpha.core.const import hs300
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
                            ui.zone("chart", size="65%"),
                            ui.zone("metrics", size="15%"),
                        ],
                    ),
                ],
                height="100vh",
            )
        ],
    )
    q.client.layout = "#bt"
    q.client.bt_progress = []
    if q.user.backtest is None:
        q.user.backtest = Expando({"baseline": hs300})


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
    q.client.history = {
        item["name"]: item for item in infos if item.get("last_trade") is not None
    }

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
    for name, alias, desc, *_ in all_strategies:
        strategy_items.append(
            ui.expander(
                name,
                label=capitalize_first_letter(name),
                items=[
                    ui.text_xs(f"{alias}:{desc}"),
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
    bars = await Stock.get_bars_in_range(
        baseline,
        FrameType.DAY,
        start,
        end,
    )
    equities = [
        (bar["frame"], "基准", bar["close"].item() / bars[0]["open"]) for bar in bars
    ]

    return np.array(equities, dtype=[("frame", "O"), ("name", "O"), ("value", "f4")])


async def display_metrics(q: Q, metrics: dict):
    """显示回测结果的指标

    Args:
        q: Query对象
        metrics: 回测结果的指标
    """
    items = [
        ui.text(f"起始时间: {metrics['start']}"),
        ui.text(f"结束时间: {metrics['end']}"),
    ]
    labels = {
        "window": "回测窗口",
        "total_tx": "配对交易次数",
        "total_profit": "总盈亏",
        "total_profit_rate": "总盈亏率",
        "win_rate": "胜率",
        "mean_return": "笔回报收益率",
        "sharpe": "夏普比率",
        "max_drawdown": "最大回撤",
        "sortino": "Sortino比率",
        "calmar": "Calmar比率",
        "volatility": "波动率",
        "annual_return": "年化收益",
    }

    if "window" in metrics:
        items.append(ui.text(f"{labels['window']}: {metrics['window']}"))
    else:
        items.append(ui.text(f"{labels['window']}: N/A"))

    if "total_tx" in metrics:
        items.append(ui.text(f"{labels['total_tx']}: {metrics['total_tx']}"))
    else:
        items.append(ui.text(f"{labels['total_tx']}: N/A"))

    if "total_profit" in metrics:
        items.append(
            ui.text(f"{labels['total_profit']}: {metrics['total_profit']:.2f}")
        )
    else:
        items.append(ui.text(f"{labels['total_profit']}: N/A"))

    if "total_profit_rate" in metrics:
        items.append(
            ui.text(
                f"{labels['total_profit_rate']}: {metrics['total_profit_rate']:.1%}"
            )
        )
    else:
        items.append(ui.text(f"{labels['total_profit_rate']}: N/A"))

    if "win_rate" in metrics:
        items.append(ui.text(f"{labels['win_rate']}: {metrics['win_rate']:.1%}"))
    else:
        items.append(ui.text(f"{labels['win_rate']}: N/A"))

    if "mean_return" in metrics:
        items.append(ui.text(f"{labels['mean_return']}: {metrics['mean_return']:.1%}"))
    else:
        items.append(ui.text(f"{labels['mean_return']}: N/A"))

    if "sharpe" in metrics:
        items.append(ui.text(f"{labels['sharpe']}: {metrics['sharpe']:.1f}"))
    else:
        items.append(ui.text(f"{labels['sharpe']}: N/A"))

    if "max_drawdown" in metrics:
        items.append(
            ui.text(f"{labels['max_drawdown']}: {metrics['max_drawdown']:.1%}")
        )
    else:
        items.append(ui.text(f"{labels['max_drawdown']}: N/A"))

    if "sortino" in metrics:
        items.append(ui.text(f"{labels['sortino']}: {metrics['sortino']:.1f}"))
    else:
        items.append(ui.text(f"{labels['sortino']}: N/A"))

    if "calmar" in metrics:
        items.append(ui.text(f"{labels['calmar']}: {metrics['calmar']:.1f}"))
    else:
        items.append(ui.text(f"{labels['calmar']}: N/A"))

    if "volatility" in metrics:
        items.append(ui.text(f"{labels['volatility']}: {metrics['volatility']:.1f}"))
    else:
        items.append(ui.text(f"{labels['volatility']}: N/A"))

    if "annual_return" in metrics:
        items.append(
            ui.text(f"{labels['annual_return']}: {metrics['annual_return']:.1%}")
        )
    else:
        items.append(ui.text(f"{labels['annual_return']}: N/A"))

    q.page["metrics"] = ui.form_card("metrics", items=items)

    await q.page.save()


async def live_update(q: Q, start: datetime.date, end: datetime.date, event: DataEvent):
    """绘制正在回测中的资产图

    Args:
        q: Query对象
        start: 回测开始时间
        end: 回测账户结束时间
        event: DataEvent对象,当它被激活时，我们将收到回测进度数据
    """
    baseline = q.user.backtest.baseline or hs300
    baseline = await get_baseline_assets(baseline, start, end)
    q.page["chart"] = ui.plot_card(
        box="chart",
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

    while True:
        try:
            await event.wait()
            event.clear()
            msg = event.data
            logger.debug("received msg:%s", msg)
            if "sell" in msg:
                for trade in msg["sell"]:
                    order_time = arrow.get(trade["time"]).date()
                    label = f"卖出{trade['security']} {trade['price']} {trade['filled']}"
                    q.page["chart"].plot.marks.append(
                        ui.mark(x=order_time, label=label, type="point")
                    )
                await q.page.save()
                continue
            if "buy" in msg:
                trade = msg["buy"]
                order_time = arrow.get(trade["time"]).date()
                label = f"卖出{trade['security']} {trade['price']} {trade['filled']}"
                q.page["chart"].plot.marks.append(
                    ui.mark(x=order_time, label=label, type="point")
                )
            if msg["event"] == "finished":
                logger.info("backtest finished")
                break
            elif msg["event"] == "progress":
                frame = msg["frame"]

                pos = np.argwhere(baseline["frame"] == frame).flatten()
                assert len(pos) > 0
                i = pos[0]

                q.page["chart"].data[-1] = [
                    baseline[i][0].isoformat(),
                    baseline[i][1],
                    baseline[i][2].item(),
                ]
                q.page["chart"].data[-1] = [
                    frame.isoformat(),
                    hs300,
                    round(msg["info"]["assets"] / msg["info"]["principal"], 3),
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
            ui.textbox("start_date", label="开始日期", value="2022-01-04"),
            ui.textbox("end_date", label="结束日期", value="2022-06-01"),
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

    params = json.loads(q.args.params or "{}")

    q.page["meta"].dialog = None
    s = create_strategy_by_name(strategy)
    await display_metrics(q, {"start": start, "end": end})
    await q.page.save()

    try:
        token = uuid.uuid4().hex
        name = f"{s.name}-{s.version}-{token[-4:]}"
        event = await run_backtest_remote(
            strategy, start, end, principal, params, name, token
        )

        await live_update(q, start, end, event)
        client = TraderClient(cfg.backtest.url, name, token, is_backtest=False)
        metrics = client.metrics(baseline=q.user.backtest.baseline)
        await display_metrics(q, metrics)
    except Exception as e:
        # since this might be a long run, we use notification to show the error
        notify(q, f"backtest {name} failed")
        logger.exception(e)

    await q.page.save()
    raise StopPropagation


@on()
async def plot_history(q: Q):
    account = q.args.plot_history

    info = q.client.history.get(account)
    start = info["start"]
    end = info.get("last_trade")
    baseline = q.user.backtest.baseline or hs300

    token = q.client.accounts.get(account)
    client = TraderClient(cfg.backtest.url, info["name"], token, is_backtest=False)
    bills = client.bills()

    y = {
        arrow.get(item[0]).date(): item[1] / info["principal"]
        for item in bills["assets"]
    }

    assets = map(
        lambda x: (x[0], account, round(x[1] / info["principal"], 2)), bills["assets"]
    )

    curve_data = [item for item in assets]

    baseline_assets = await get_baseline_assets(baseline, start, end)
    baseline_assets = map(
        lambda x: (x[0].isoformat(), x[1], round(x[2].item(), 2)), baseline_assets
    )
    curve_data.extend(baseline_assets)

    # show buy/sell
    trade_marks = []
    for trade in bills["trades"].values():
        order_date = arrow.get(trade["time"]).date()
        label = f"{trade['order_side']}\n{trade['security']}\n成交量:{trade['filled']}\n成交价:{trade['price']}"
        trade_marks.append(
            ui.mark(
                x=order_date.isoformat(), y=y.get(order_date), label=label, type="point"
            )
        )

    q.page["chart"] = ui.plot_card(
        box="chart",
        title="Line, groups",
        data=data("date series assets", len(curve_data), curve_data, pack=True),
        plot=ui.plot(
            [
                ui.mark(
                    type="line",
                    x="=date",
                    y="=assets",
                    color="series",
                    x_scale="time-category",
                ),
                *trade_marks,
            ]
        ),
    )

    metrics = client.metrics(baseline=q.user.backtest.baseline)
    await display_metrics(q, metrics)
