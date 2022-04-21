from flask import session
from alpha.web import routing
from alpha.web.auth.models import sessions
from alpha.web.views.layout import with_header
import dash_bootstrap_components as dbc
import dash
from dash import Input, Output, State, callback, html, dcc
import arrow
from omicron import tf
from omicron.models.stock import Stock
from alpha.web.utils import get_bars, get_triggerred_controls, make_stock_input_hint
from coretypes import Frame, FrameType
import plotly.graph_objects as go

param_keys = ("start", "end", "baseline", "capital")


def prefix(sub: str):
    assert sub in param_keys

    return f"BACKTEST_{sub.uppper()}"


toolbar_style = {
    "display": "flex",
    "justify-content": "end",
    "align-items": "center",
    "height": "2rem",
    "width": "50vw",
}


def toolbar():
    end = arrow.now()
    start = end.shift(days=-365)

    label = html.P("策略名:", style={"margin-right": "1rem"})
    title = html.Div(
        [label, html.P("网格交易", id="strategy-name")],
        style={"display": "flex", "flex-direction": "row"},
    )
    toolbar = html.Div(
        [
            dbc.InputGroup(
                [
                    dbc.InputGroupText(
                        "回测区间", style={"margin-right": "0.5rem", "height": "2rem"}
                    ),
                    dbc.Input(
                        id="start-date-input",
                        type="date",
                        value=start.format("YYYY-MM-DD"),
                    ),
                    dbc.InputGroupText("-", style={"margin": "0 0.5rem"}),
                    dbc.Input(
                        id="end-date-input", type="date", value=end.format("YYYY-MM-DD")
                    ),
                ],
                size="sm",
                style={"width": "20rem"},
            ),
            dbc.InputGroup(
                [
                    dbc.InputGroupText(
                        "比较基准", style={"margin-right": "0.5rem", "height": "2rem"}
                    ),
                    dbc.Input(
                        placeholder="沪深300",
                        list="stock-index-hints",
                        id="baseline-input",
                        style={"height": "2rem", "width": "6rem"},
                        debounce=True,
                        size="sm",
                    ),
                    html.Datalist(id="stock-index-hints"),
                ],
                size="sm",
                style={"width": "14rem"},
            ),
            dbc.ButtonGroup(
                [
                    dbc.Button(
                        [
                            html.I(className="fas fa-caret-right"),
                            html.Span("开始回测", style={"margin-left": "0.5rem"}),
                        ],
                        color="secondary",
                        size="sm",
                        title="向后一天",
                        id="backtest",
                        style={"width": "6rem"},
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

    return html.Div(
        [title, toolbar],
        style={
            "display": "flex",
            "justify-content": "space-between",
        },
    )

def validate_baseline_code(baseline: str):
    try:
        Stock(baseline)
        sessions.save(prefix("baseline"), baseline)
        return True
    except Exception:
        return False


def make_response(**kwargs):
    defaults = {
        "start": dash.no_update,
        "end": dash.no_update,
        "baseline": dash.no_update,
    }

    for k in param_keys:
        defaults[k] = sessions.get(prefix(k))

    defaults.update(kwargs)

    return defaults


def get_params():
    end = sessions.get(prefix("end"), arrow.now())
    start = sessions.get(prefix("start"), end.shift(days=-365))
    baseline = sessions.get(prefix("baseline"), "399300.XSHE")
    capital = sessions.get(prefix("capital"), 1_000_000)

    return start.date(), end.date(), baseline, capital


@callback(Output("stock-index-hints", "children"), [Input("baseline-input", "value")])
def update_stock_index_hints(code: str):
    if not code:
        return []

    return make_stock_input_hint(code)


@callback(
    Output("canvas-container", "children"),
    [Input("backtest", "n_clicks")],
    [
        State("start-date-input", "value"),
        State("end-date-input", "value"),
        State("baseline-input", "value"),
    ],
    prevent_initial_call=True,
)
def draw_backtest_result(_, start, end, baseline):
    triggered = get_triggerred_controls()
    if "start-date-input" in triggered():
        start = arrow.get(start).date()
        sessions.save(prefix("start"), start)

    if "end-date-input" in triggered():
        end = arrow.get(end).date()
        sessions.save(prefix("end"), end)

    if "baseline-input" in triggered():
        if not validate_baseline_code(baseline):
            return make_response(baseline_input_invalid=True)


def draw_baseline():
    start, end, baseline, capital = get_params()

    n = tf.count_day_frames(start, end)
    bars = get_bars(baseline, end, n, FrameType.DAY)
    close = bars["close"]
    equity = close /close[0] * capital
    
    ticks = [tf.date2int(x) for x in bars["frame"]]
    return dcc.Graph(go.Scatter(x=ticks, y=equity, markers="line"))

@routing.dispatch("/backtest")
def init_page():
    main = dbc.Container(
        [
            dbc.Row(toolbar()),
            dbc.Row(id="canvas-container", style={"height": "100vh"}),
        ],
        fluid=True,
    )
    return with_header(main)
