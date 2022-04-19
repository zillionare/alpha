import logging
from typing import List, Union
from dash.development.base_component import Component

import arrow
import dash
import dash_bootstrap_components as dbc
from asgiref.sync import async_to_sync
from coretypes import Frame, FrameType, SecurityType
from dash import Input, Output, callback, dcc, html
from omicron.models.stock import Stock
from collections import OrderedDict

from pandas_datareader import get_recent_iex
from alpha.web.utils import get_triggerred_controls
from dateutil import parser
from alpha.web.components.scaffold import render_with_scaffold
from alpha.web import session

from alpha.plotting.candlestick import Candlestick
from alpha.web import routing
import re

logger = logging.getLogger(__name__)

_state_keys = ("code", "dt", "nbars", "bbox_size", "use_close", "sr_win")


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
            "alert": dash.no_update,
        }
    )

    response.update(kwargs)

    return list(response.values())


def validate_code(sid, code):
    if code is None:
        return True

    try:
        _ = Stock(code)
        session.save(sid, _key("code"), code)
        return True
    except Exception:
        return False


def validate_date(sid: str, date: str):
    if date is None:
        return True

    try:
        dt = parser.parse(date)
        session.save(sid, _key("dt"), dt)
        return True
    except Exception:
        return False


@callback(
    Output("main-fig", "figure"),
    Output("stock-input", "value"),
    Output("stock-input", "n_submit"),
    Output("stock-input", "invalid"),
    Output("date-input", "invalid"),
    Output("alert", "children"),
    Input("session_id", "data"),
    Input("stock-input", "value"),
    Input("date-input", "value"),
    prevent_initial_call=True
)
def on_params_change(sid: str, code, date):
    triggered = get_triggerred_controls()

    control_state = {}
    if "stock-input" in triggered:
        if not validate_code(sid, code):
            return _update_toolbar(stock_input_invalid=True)
        else:
            control_state["stock_input_invalid"] = False

    if "date-input" in triggered:
        if not validate_date(sid, date):
            return _update_toolbar(date_input_invalid=True)
        else:
            control_state["date_input_invalid"] = False

    figure = make_main_figure(sid)
    return _update_toolbar(figure=figure, **control_state)


@async_to_sync
async def get_bars(
    stock: str, end: Frame, n: int, frame_type: FrameType, fq=True, unclosed=True
):
    return await Stock.get_bars(stock, n, frame_type, end, fq, unclosed)


def load_params(sid: str, **kwargs) -> dict:
    params = {
        "dt": arrow.now().date(),
        "code": "000001.XSHG",
        "nbars": 120,
        "bbox_size": 20,
        "use_close": True,
        "sr_win": 60,  # support_resist_lines win
    }

    for field in _state_keys:
        v = session.get(sid, _key(field))
        if v is not None:
            params[field] = v

        if field in kwargs:
            params[field] = kwargs.get(field)

    return params


def make_main_figure(sid: str, **kwargs):
    params = load_params(sid, **kwargs)

    code = params["code"]
    bars = get_bars(params["code"], params["dt"], params["nbars"], FrameType.DAY)

    stock = Stock(code)
    name = stock.display_name

    logger.info("draw candlestick for %s at %s", code, params["dt"])
    # build figure
    cs = Candlestick(bars, title=name)
    cs.mark_bbox(min_size=params["bbox_size"])

    # if stock.security_type == SecurityType.INDEX:
    #     up_thres = 0.01
    #     down_thres = -0.01
    # else:
    #     up_thres = 0.03
    #     down_thres = -0.03

    cs.mark_peaks_and_valleys()
    cs.mark_support_resist_lines(use_close=params["use_close"], win=params["sr_win"])

    return cs.figure


@routing.on("/research")
def init_page(sid: str):
    figure = make_main_figure(sid)
    graph = dcc.Graph(id="main-fig", figure=figure)
    return render_view(graph)


def render_view(components: Union[Component, List[Component]]):
    if not isinstance(components, list):
        components = [components]

    content = dbc.Container(
        [
            dbc.Alert(
                id="alert",
                dismissable=True,
                is_open=False,
                color="danger",
                style={"height": "3rem"},
            ),
            toolbar(),
            *components,
        ],
        fluid=True,
        style={"padding": 0},
    )
    location = dcc.Location("researchUrl", pathname="/research", refresh=False)
    return render_with_scaffold(location, content)
