import datetime
import logging

import arrow
from coretypes import Frame, FrameType
from h2o_wave import Expando, Q, ui
from omicron import tf
from omicron.models.stock import Stock
from plotly import io as pio

from alpha.plotting.candlestick import Candlestick
from alpha.web.utils import inlinejs, make_stock_input_hint
from alpha.web.widgets import header

from ..routing import StopPropagation, handle_on, on

logger = logging.getLogger(__name__)

# todo: remove this
symbol_input_selector = "[data-test=change_symbol] input"
page_script = f"""
    var selector = "{symbol_input_selector}";
    var event = "input";
    var callback = wave_emit("change_symbol", "on_symbol_hint", "value");
    
    bind_event(selector, event, callback);
    console.info("bind event to:", selector, event);
"""


def left_panel(q: Q):
    rows = []
    for code in q.user.research.favorites:
        rows.append(
            ui.table_row(
                name=code, cells=[Stock(code).display_name, code.split(".")[0]]
            )
        )
    return ui.form_card(
        "left",
        items=[
            ui.inline(
                items=[
                    ui.combobox(
                        name="change_symbol",
                        choices=["000001.XSHG 上证综指", "399001.XSHE 深证成指", "399006.XSHE 创业板指"],
                        value=q.client.research.choosed_symbol,
                        width="80%",
                        trigger=True,
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
                rows=rows
            ),
        ],
    )


def right_panel():
    # Render the plot as an HTML.
    return ui.MarkdownCard("right", title="诊股", content="right panel")


async def content(q: Q):
    code = q.client.research.symbol_choosed or q.user.research.code
    n = q.user.research.nbars
    frame_type = q.user.research.frame_type
    end = q.client.research.end

    bars = await Stock.get_bars(code, n, frame_type, end, unclosed=True)
    name = Stock(code).display_name
    frame = {
        FrameType.MIN1: "1分钟",
        FrameType.MIN30: "30分钟",
        FrameType.MIN60: "60分钟",
        FrameType.DAY: "日线",
        FrameType.WEEK: "周线",
        FrameType.MONTH: "月线",
    }.get(frame_type)

    cs: Candlestick = Candlestick(bars, title=f"{name} {frame}")
    if q.user.research.display_trendline:
        cs.mark_support_resist_lines()

    if q.user.research.display_bbox:
        cs.mark_bbox()

    if q.user.research.display_pv:
        cs.mark_peaks_and_valleys()

    # config = {"scrollZoom": False, "showLink": False, "displayModeBar": False}
    html = pio.to_html(cs.figure, validate=False, include_plotlyjs="cdn")
    graph = ui.frame(content=html, height="90vh", width="100%")

    return ui.form_card("content", items=[toolbar(q), graph])


def _format_time_textbox(time: Frame):
    logger.info("time is: %s", time)
    if type(time) == datetime.datetime:
        return time.strftime("%y-%m-%d %H:%M")
    else:
        return time.strftime("%Y-%m-%d")


def toolbar(q: Q):
    start = q.client.research.start
    end = q.client.research.end

    logger.info("start: %s, end: %s", start, end)
    if type(end) == datetime.datetime:
        disable_forward = end >= arrow.now().naive
    else:
        disable_forward = end >= tf.day_shift(arrow.now(), 0)

    return ui.inline(
        items=[
            ui.text("开始"),
            ui.textbox(
                "set_start",
                width="10vw",
                value=_format_time_textbox(start),
                trigger=True,
            ),
            ui.text("结束"),
            ui.textbox(
                "set_end", width="10vw", value=_format_time_textbox(end), trigger=True
            ),
            ui.button("fastbackward", caption="后退30", icon="Rewind"),
            ui.button("backward", caption="后退一", icon="PlayReverse"),
            ui.button("forward", caption="前进一", icon="Play", disabled=disable_forward),
            ui.button(
                "fastforward",
                caption="前进30",
                icon="FastForward",
                disabled=disable_forward,
            ),
            ui.menu(
                [
                    ui.command("set_frame_type_1d", label="日线", icon="Calendar"),
                    ui.command("set_frame_type_1w", label="周线", icon="Calendar"),
                    ui.command("set_frame_type_1M", label="月线", icon="Calendar"),
                    ui.command("set_frame_type_30m", label="30分钟线", icon="Calendar"),
                    ui.command("set_frame_type_1m", label="分钟线", icon="Calendar"),
                ],
                name="frame_type",
                icon="CalendarSettings",
            ),
            ui.button("settings", caption="设置", icon="Settings", primary=False),
        ],
        justify="end",
        inset=True,
    )


@on("settings")
async def settings_dialog(q: Q):
    """设置对话框"""
    cfg = q.user.research

    q.page["meta"].dialog = ui.dialog(
        title="设置",
        name="settings_dialog",
        items=[
            ui.checkbox(
                "set_display_trendline", label="显示趋势线", value=cfg.display_trendline
            ),
            ui.checkbox(
                "set_display_bbox", value=cfg.display_bbox, label="是否显示Bounding box"
            ),
            ui.checkbox("set_display_pv", value=cfg.display_pv, label="是否标示峰谷"),
            ui.textbox("set_nbars", value=str(cfg.nbars), label="每屏显示的K线数量"),
            ui.textbox("set_bbox_win", label="平台检测的窗口大小", value=str(cfg.bbox_win)),
            ui.textbox(
                "set_pv_thresh", label="峰谷检测相对高度阈值", value=f"{cfg.pv_thresh:.1%}"
            ),
            ui.buttons(
                items=[
                    ui.button("save_settings", caption="确定", primary=True),
                    ui.button("cancel_dialog", caption="取消"),
                ],
                justify="end",
            ),
        ],
        blocking=True,
        closable=True,
    )
    await q.page.save()


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
    await render_view(q)


def init(q: Q):
    q.client.initialized = True
    q.client.research = Expando(
        {
            "end": tf.day_shift(arrow.now(), 0),
            "start": tf.day_shift(arrow.now(), -120),
            "symbol_list": ["000001 - 上证综指", "399001 - 深证成指", "399006 - 创业板指"],
            "symbol_choosed": "",
        }
    )

    if not q.user.research:
        q.user.research = Expando(
            {
                "code": "000001.XSHG",
                "nbars": 120,
                "bbox_win": 20,
                "pv_thresh": 0.03,
                "display_trendline": False,
                "display_bbox": False,
                "frame_type": FrameType.DAY,
                "favorites": {"000001.XSHG", "399001.XSHE", "399006.XSHE"},
            }
        )

    q.client.research.symbo_choosed = q.user.research.code


async def render_view(q: Q):
    if not q.client.initialized:
        init(q)

    # todo: this will drop all cards thus degrade performance, could be just update different cards
    q.page.drop()

    logger.info(
        "render candlestick for %s, from %s to %s, %s",
        q.user.research.code,
        q.client.research.start,
        q.client.research.end,
        q.user.research.frame_type,
    )
    set_layout(q)
    q.page["header"] = header("research")
    q.page["left"] = left_panel(q)
    q.page["right"] = right_panel()
    q.page["content"] = await content(q)

    await q.page.save()


@on()
async def change_symbol(q: Q):
    code = q.args.change_symbol.split(" ")[0]
    logger.info("change symbol to %s", q.args.change_symbol)
    matched = Stock.fuzzy_match(code)
    if len(matched) == 1:
        code = list(matched.keys())[0]
        q.client.research.choosed_symbol = code
    else:
        return

    q.client.research.symbol_choosed = code
    await render_view(q)


@on()
async def set_frame_type_1d(q: Q):
    q.user.research.frame_type = FrameType("1d")
    await render_view(q)


@on()
async def set_frame_type_1w(q: Q):
    q.user.research.frame_type = FrameType("1w")
    await render_view(q)


@on()
async def set_frame_type_1M(q: Q):
    q.user.research.frame_type = FrameType("1M")
    await render_view(q)


@on()
async def set_frame_type_30m(q: Q):
    q.user.research.frame_type = FrameType("30m")
    end = q.client.research.end

    if type(end) == datetime.date:
        end = min(tf.combine_time(end, 15), tf.floor(arrow.now().naive, FrameType.MIN1))
        start = tf.shift(
            tf.floor(end, FrameType.MIN30),
            q.user.research.nbars * -1,
            FrameType.MIN30,
        )
    else:
        close_end = tf.floor(end, FrameType.MIN30)
        start = tf.shift(close_end, q.user.research.nbars * -1, FrameType.MIN30)

    q.client.research.end = end
    q.client.research.start = start

    await render_view(q)


@on()
async def set_frame_type_1m(q: Q):
    q.user.research.frame_type = FrameType("1m")
    end = q.client.research.end

    if type(end) == datetime.date:
        end = min(tf.combine_time(end, 15), tf.floor(arrow.now().naive, FrameType.MIN1))
        start = tf.shift(end, q.user.research.nbars * -1, FrameType.MIN1)
    else:
        close_end = tf.floor(end, FrameType.MIN1)
        start = tf.shift(close_end, q.user.research.nbars * -1, FrameType.MIN1)

    q.client.research.end = end
    q.client.research.start = start

    await render_view(q)


@on()
async def fastbackward(q: Q):
    await on_forward_backward(q, -30)


@on()
async def backward(q: Q):
    await on_forward_backward(q, -1)


@on()
async def forward(q: Q):
    await on_forward_backward(q, 1)


@on()
async def fastforward(q: Q):
    await on_forward_backward(q, 30)


async def on_forward_backward(q: Q, span: int):
    frame_type = q.user.research.frame_type

    if frame_type in tf.minute_level_frames:
        end = tf.floor(q.client.research.end, frame_type)
    else:
        end = tf.day_shift(q.client.research.end, 0)

    q.client.research.end = tf.shift(end, span, frame_type)
    q.client.research.start = tf.shift(end, -q.user.research.nbars, frame_type)

    await render_view(q)


@on()
async def set_start(q: Q):
    frame_type = q.user.research.frame_type

    try:
        start = arrow.get(q.args.set_start)
    except Exception:
        # user input is not a date/datetime
        return

    if frame_type in tf.minute_level_frames:
        if start.hour == 0:
            # for minute level frame, we need hour and minute
            return
        else:
            start = start.naive
    else:
        start = start.date()

    if start == q.client.research.start:
        return

    q.client.research.start = start

    await render_view(q)


@on()
async def set_end(q: Q):
    frame_type = q.user.research.frame_type

    try:
        end = arrow.get(q.args.set_end)
    except Exception:
        return

    if frame_type in tf.minute_level_frames:
        if end.hour == 0:
            # for minute level frame, we need hour and minute
            return
        else:
            end = end.naive
    else:
        end = end.date()

    if end == q.client.research.end:
        return

    q.client.research.end = end
    await render_view(q)


@on()
async def save_settings(q: Q):
    cfg = q.user.research

    cfg.display_trendline = q.args.set_display_trendline
    cfg.display_bbox = q.args.set_display_bbox
    cfg.display_pv = q.args.set_display_pv

    try:
        cfg.bbox_win = int(q.args.set_bbox_win)
        cfg.pv_thresh = float(q.args.set_pv_thresh[:-1]) * 0.01
    except Exception as e:
        logger.info("invalid settings: %s, %s", q.args.bbox_win, q.args.pv_thresh)
        logger.exception(e)

    await render_view(q)

@on()
async def add_favorite(q: Q):
    code = q.client.research.symbol_choosed

    if code not in q.user.research.favorites:
        q.user.research.favorites.add(code)
        await render_view(q)

@on("favorites")
async def on_click_favorite(q: Q):
    q.client.research.symbol_choosed = q.args.favorites[0]
    await render_view(q)
