import datetime
import logging
import re
from collections import OrderedDict
from typing import List, Union

import arrow
import dash
import dash_bootstrap_components as dbc
from asgiref.sync import async_to_sync
from coretypes import Frame, FrameType, SecurityType
from dash import Input, Output, callback, dcc, html
from dash.development.base_component import Component
from dash_bootstrap_components import (
    Button,
    ButtonGroup,
    Checkbox,
    FormText,
    InputGroup,
    InputGroupText,
    Modal,
    ModalBody,
    ModalFooter,
    ModalHeader,
    ModalTitle,
)
from dateutil import parser
from omicron import tf
from omicron.models.stock import Stock

from alpha.plotting.candlestick import Candlestick
from alpha.web import routing
from alpha.web.auth.models import sessions
from alpha.web.components.scaffold import render_with_scaffold
from alpha.web.components.widgets import make_form
from alpha.web.utils import get_triggerred_controls

logger = logging.getLogger(__name__)

_state_keys = (
    "code",
    "dt",
    "nbars",
    "bbox_size",
    "use_close",
    "sr_win",
    "error",
    "frame_type",
)


def _key(sub: str) -> str:
    assert sub in _state_keys

    return f"RESEARCH.{sub.upper}"


toolbar_style = {
    "display": "flex",
    "justify-content": "end",  # 右对齐
    "align-items": "center",
    "height": "2rem",
    "width": "100%",
}


def toolbar():
    now = arrow.now().format("YYYY-MM-DD")
    toolbar = html.Div(
        [
            dbc.Input(
                id="stock-input",
                list="stock-hints",
                type="search",
                placeholder="代码或名称",
                html_size="8",
                size="sm",
                n_submit=0,
                style={"margin-right": "1rem", "width": "8rem", "height": "2rem"},
            ),
            html.Datalist(id="stock-hints"),
            dbc.InputGroup(
                [
                    dbc.InputGroupText(
                        "日期", style={"margin-right": "0.5rem", "height": "2rem"}
                    ),
                    dbc.Input(
                        placeholder=now,
                        id="date-input",
                        style={"height": "2rem"},
                        debounce=True,
                    ),
                ],
                style={"margin-right": "1rem", "width": "10rem"},
            ),
            dbc.ButtonGroup(
                [
                    dbc.Button(
                        html.I("30", className="fas auto"), size="sm", id="frametype-30"
                    ),
                    dbc.Button(
                        html.I("日", className="fas auto"),
                        size="sm",
                        active=True,
                        id="frametype-day",
                    ),
                    dbc.Button(
                        html.I("周", className="fas auto"),
                        size="sm",
                        id="frametype-week",
                    ),
                    dbc.Button(
                        html.I(className="fas fa-backward"),
                        color="secondary",
                        size="sm",
                        style={"width": "2rem", "backgroundColor": "transparent"},
                        title="向前一月",
                        id="fastbackward",
                    ),
                    dbc.Button(
                        html.I(className="fas fa-caret-left"),
                        color="secondary",
                        size="sm",
                        style={"width": "2rem"},
                        title="向前一天",
                        id="backward",
                    ),
                    dbc.Button(
                        html.I(className="fas fa-caret-right"),
                        color="secondary",
                        size="sm",
                        style={"width": "2rem"},
                        title="向后一天",
                        id="forward",
                    ),
                    dbc.Button(
                        html.I(className="fas fa-forward"),
                        color="secondary",
                        size="sm",
                        style={"width": "2rem"},
                        title="向后一月",
                        id="fastforward",
                    ),
                    dbc.Button(
                        html.I(className="fas fa-gear"),
                        color="secondary",
                        size="sm",
                        style={"width": "2rem"},
                        title="设置",
                        id="settings",
                    ),
                ]
            ),
        ],
        style=toolbar_style,
    )

    return toolbar


@callback(
    Output(component_id="stock-hints", component_property="children"),
    [Input(component_id="stock-input", component_property="value")],
)
def update_datalist(value):
    if value is None:
        return []

    matched = Stock.fuzzy_match(value)
    #  ('000001.XSHE', '平安银行', 'PAYH'... 'stock')

    options = []

    if re.match(r"\d+", value):  # 用户输入了代码
        for v in matched.values():
            code = v[0].split(".")[0]
            options.append(html.Option(v[0], label=f"{code} {v[1]}"))
    elif re.match(r"[a-z]+", value.lower()):
        for v in matched.values():
            options.append(html.Option(v[0], label=f"{v[2]} {v[1]}"))
    else:
        for v in matched.values():
            options.append(html.Option(v[0], label=f"{v[1]}"))

    return options


def _update_toolbar(**kwargs):
    response = OrderedDict(
        {
            "figure": dash.no_update,
            "stock_value": dash.no_update,
            "stock_n_submit": dash.no_update,
            "stock_input_invalid": dash.no_update,
            "date_input_invalid": dash.no_update,
            "date_input_value": dash.no_update,
            "frametype_30_active": dash.no_update,
            "frametype_day_active": dash.no_update,
            "frametype_week_active": dash.no_update,
            "alert": dash.no_update,
        }
    )

    if sessions.get(_key("error")) is not None:
        response["alert"] = dbc.Alert(
            sessions.get(_key("error")),
            color="danger",
            dismissable=True,
            duration=1000 * 10,
        )
        sessions.delete(_key("error"))

    if "date_input_invalid" not in kwargs:  # 可能按了快进，需要更新日期框
        response["date_input_value"] = sessions.get(_key("dt"))

    response.update(kwargs)

    return list(response.values())


def validate_code(code):
    if code is None:
        return True

    try:
        _ = Stock(code)
        sessions.save(_key("code"), code)
        return True
    except Exception:
        return False


def validate_date(date: str):
    if date is None:
        return True

    try:
        dt = parser.parse(date)
        sessions.save(_key("dt"), dt)
        return True
    except Exception:
        return False


@callback(
    Output("main-fig", "figure"),
    Output("stock-input", "value"),
    Output("stock-input", "n_submit"),
    Output("stock-input", "invalid"),
    Output("date-input", "invalid"),
    Output("date-input", "value"),
    Output("frametype-30", "active"),
    Output("frametype-day", "active"),
    Output("frametype-week", "active"),
    Output("alert", "children"),
    Input("stock-input", "value"),
    Input("date-input", "value"),
    Input("frametype-30", "n_clicks"),
    Input("frametype-day", "n_clicks"),
    Input("frametype-week", "n_clicks"),
    Input("fastbackward", "n_clicks"),
    Input("backward", "n_clicks"),
    Input("forward", "n_clicks"),
    Input("fastforward", "n_clicks"),
    prevent_initial_call=True,
)
def on_params_change(code, date, f30, fday, fweek, fb, b, f, ff):
    triggered = get_triggerred_controls()

    control_state = {}
    if "stock-input" in triggered:
        if not validate_code(code):
            return _update_toolbar(stock_input_invalid=True)
        else:
            control_state["stock_input_invalid"] = False

    if "date-input" in triggered:
        if not validate_date(date):
            return _update_toolbar(date_input_invalid=True)
        else:
            control_state["date_input_invalid"] = False

    dt = sessions.get(_key("dt")) or arrow.now().date()
    if "fastbackward" in triggered:
        # todo: 如何设置正确的最小日期？
        dt = max(tf.day_shift(dt, -30), datetime.date(2015, 1, 4))

    elif "backward" in triggered:
        dt = max(tf.day_shift(dt, -1), datetime.date(2015, 1, 4))

    elif "forward" in triggered:
        dt = min(tf.day_shift(dt, 1), arrow.now().date())

    elif "fastforward" in triggered:
        dt = min(tf.day_shift(dt, 30), arrow.now().date())

    sessions.save(_key("dt"), dt)

    if "frametype-30" in triggered:
        sessions.save(_key("frame_type"), FrameType.MIN30)
        control_state["frametype_30_active"] = True
        control_state["frametype_day_active"] = False
        control_state["frametype_week_active"] = False
    elif "frametype-day" in triggered:
        sessions.save(_key("frame_type"), FrameType.DAY)
        control_state["frametype_30_active"] = False
        control_state["frametype_day_active"] = True
        control_state["frametype_week_active"] = False
    else:
        control_state["frametype_30_active"] = False
        control_state["frametype_day_active"] = False
        control_state["frametype_week_active"] = True
        sessions.save(_key("frame_type"), FrameType.WEEK)

    figure = make_main_figure()
    return _update_toolbar(figure=figure, **control_state)


@async_to_sync
async def get_bars(
    stock: str, end: Frame, n: int, frame_type: FrameType, fq=True, unclosed=True
):
    return await Stock.get_bars(stock, n, frame_type, end, fq, unclosed)


def load_params(**kwargs) -> dict:
    params = {
        "dt": arrow.now().date(),
        "code": "000001.XSHG",
        "nbars": 120,
        "bbox_size": 20,
        "use_close": True,
        "sr_win": 60,  # support_resist_lines win
        "frame_type": FrameType.DAY,
    }

    for field in _state_keys:
        v = sessions.get(_key(field))
        if v is not None:
            params[field] = v

        if field in kwargs:
            params[field] = kwargs.get(field)

    return params


def make_main_figure(**kwargs):
    params = load_params(**kwargs)

    code = params["code"]
    frame_type = params["frame_type"]
    if frame_type == FrameType.MIN30:
        frame_desc = "30分钟线"
    elif frame_type == FrameType.DAY:
        frame_desc = "日线"
    else:
        frame_desc = "周线"

    bars = get_bars(params["code"], params["dt"], params["nbars"], frame_type)

    stock = Stock(code)
    name = stock.display_name

    logger.info("draw candlestick for %s at %s", code, params["dt"])
    # build figure
    cs = Candlestick(bars, title=f"{name} - {frame_desc}")

    cs.mark_bbox(min_size=params["bbox_size"])

    if stock.security_type == SecurityType.INDEX:
        up_thres = 0.01
        down_thres = -0.01
    else:
        up_thres = 0.03
        down_thres = -0.03

    cs.mark_peaks_and_valleys()

    cs.mark_support_resist_lines(
        up_thres, down_thres, use_close=params["use_close"], win=params["sr_win"]
    )

    return cs.figure


@callback(Output("settings-container", "children"), Input("settings", "n_clicks"))
def on_settings_click(n):
    if "settings" not in get_triggerred_controls():
        return dash.no_update

    # "nbars", "bbox_size", "use_close", "sr_win"
    labels = ["K线数", "平台宽度", "支撑线", "峰谷检测"]
    types = ["text", "text", "checkbox", "text"]

    nbars = sessions.get(_key("nbars"), 120)
    bbox_size = sessions.get(_key("bbox_size"), 20)
    use_close = sessions.get(_key("use_close"), True)
    sr_win = sessions.get(_key("sr_win"), 60)

    tooltips = [
        "窗口内显示的K线数",
        "最小宽度，小于此数值时不认为是平台",
        f"支撑线/阻力线数据源，当前使用{'close' if use_close else 'high/low'}",
        f"在多少个周期内检测峰谷",
    ]

    icons = ["fas fa-home"] * len(labels)
    defaults = [nbars, bbox_size, use_close, sr_win]

    return dbc.Modal(
        [
            ModalHeader(ModalTitle("param settings")),
            ModalBody(
                make_form("settings-input", labels, types, defaults, tooltips, icons)
            ),
            ModalFooter(
                [
                    Button(
                        id="settings-save", children="save", color="primary", size="sm"
                    ),
                    Button(
                        id="settings-cancel",
                        children="cancel",
                        color="secondary",
                        size="sm",
                    ),
                ]
            ),
        ],
        id="settings-modal",
        is_open=True,
        size="mb",
    )


@callback(
    Output("router", "pathname"),
    Input("settings-save", "n_clicks"),
    Input("settings-input-0", "value"),
    Input("settings-input-1", "value"),
    Input("settings-input-2", "value"),
    Input("settings-input-3", "value"),
    prevent_initial_call=True,
)
def on_settings_save(nclicks, nbars, bbox_size, use_close, sr_win):
    if "settings-save" in get_triggerred_controls() and nclicks:
        # save settings
        if nbars:
            sessions.save(_key("nbars"), int(nbars))

        if bbox_size:
            sessions.save(_key("bbox_size"), int(bbox_size))

        if use_close is not None:
            sessions.save(_key("use_close"), use_close)

        if sr_win:
            sessions.save(_key("sr_win"), int(sr_win))

        return "/research"
    else:
        return dash.no_update


@callback(
    Output("settings-modal", "is_open"),
    Input("settings-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def on_settings_cancel(nclicks):
    if nclicks:
        return False


@routing.on("/research")
def init_page():
    figure = make_main_figure()
    graph = dcc.Graph(id="main-fig", figure=figure)
    return render_view(graph)


def render_view(components: Union[Component, List[Component]]):
    if not isinstance(components, list):
        components = [components]

    content = dbc.Container(
        [
            html.Div(id="alert", className="alert-container"),
            html.Div(id="settings-container"),
            toolbar(),
            *components,
        ],
        fluid=True,
        style={"padding": 0},
    )
    location = dcc.Location("researchUrl", pathname="/research", refresh=False)
    return render_with_scaffold(location, content)
