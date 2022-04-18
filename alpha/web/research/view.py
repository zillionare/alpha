import logging

import arrow
import dash
import dash_bootstrap_components as dbc
from asgiref.sync import AsyncToSync, async_to_sync
from coretypes import Frame, FrameType
from dash import Input, Output, State, callback, dcc, html
from omicron.models.stock import Stock
from alpha.web.utils import get_triggerred_controls
from dateutil import parser

from alpha.plotting.candlestick import Candlestick
from alpha.web import routing
from alpha.web.components.layout import make_page
import re

logger = logging.getLogger(__name__)

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
                        placeholder=now, id="date-input", style={"height": "2rem"}, debounce=True
                    ),
                ],
                style={"margin-right": "1rem", "width": "10rem"},
            ),
            dbc.ButtonGroup(
                [
                    dbc.Button(
                        "向前",
                        color="secondary",
                        size="sm",
                        style={"width": "4rem"},
                        title="向前一个月",
                    ),
                    dbc.Button(
                        "向后",
                        color="secondary",
                        size="sm",
                        style={"width": "4rem"},
                        title="向后一个月",
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

    if re.match(r'\d+', value): # 用户输入了代码
        for v in matched.values():
            code = v[0].split(".")[0]
            options.append(html.Option(v[0], label=f"{code} {v[1]}"))
    elif re.match(r'\[a-z]+', value.lower()):
        for v in matched.values():
            options.append(html.Option(v[0], label=f"{v[2]} {v[1]}"))
    else:
        for v in matched.values():
            options.append(html.Option(v[0], label=f"{v[1]}"))

    return options


@callback(
    Output("main-fig", "figure"),
    Output("stock-input", "value"),
    Output("stock-input", "n_submit"),
    Output("stock-input", "invalid"),
    Output("date-input", "invalid"),
    Output("alert", "children"),
    Input("stock-input", "value"),
    Input("stock-input", "n_submit"),
    Input("date-input", "value")
)
def change_candlestick(code, n_stock_submit, date):
    controls = get_triggerred_controls()

    if "stock-input" in controls and (code is None or n_stock_submit == 0):
        # 用户没有提交查询，只进行模糊匹配
        return dash.no_update, dash.no_update, n_stock_submit, dash.no_update, dash.no_update, dash.no_update

    try:
        now = arrow.now().date() if date is None else parser.parse(date).date()
    except Exception as e:
        logger.exception(e)
        logger.info("Invalid date: %s", date)

        return dash.no_update, dash.no_update, 0, dash.no_update, True, dash.no_update

    try:
        stock = Stock(code)
    except Exception as e:
        logger.exception(e)
        logger.info("Invalid stock: %s", code)

        return dash.no_update, dash.no_update, 0, True, dash.no_update, dash.no_update

    try:
        stock = Stock(code)
        name = stock.display_name
        
        bars = get_bars(stock.code, now, 120, FrameType.DAY)
        cs = Candlestick(bars, title=name)

        # clear invalid state
        return cs.figure, code, 0, False, False, dash.no_update
    except Exception as e:
        logger.exception(e)
        logger.info("input: %s, %s", code, date)
        return dash.no_update, dash.no_update, 0, False, dash.no_update, str(e)


@async_to_sync
async def get_bars(
    stock: str, end: Frame, n: int, frame_type: FrameType, fq=True, unclosed=True
):
    return await Stock.get_bars(stock, n, frame_type, end, fq, unclosed)


@routing.on("/research")
def render_research_page(code: str = "000001.XSHG"):
    now = arrow.now().date()
    bars = get_bars(code, now, 120, FrameType.DAY)

    name = Stock(code).display_name

    # build figure
    cs = Candlestick(bars, title=name)
    cs.mark_bounding_box()
    cs.mark_peaks_and_valleys()

    content = dbc.Container(
        [
            dbc.Alert(id="alert", dismissable=True, is_open=False, color="danger", style={"height": "3rem"}),
            toolbar(), 
            dcc.Graph(id="main-fig", figure=cs.figure)
        ],
        fluid=True,
        style={"padding": 0},
    )
    location = dcc.Location("researchUrl", pathname="/research", refresh=False)
    return make_page(location, content)
