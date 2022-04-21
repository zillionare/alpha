from numpy import size
from alpha.web import routing
from alpha.web.views.layout import with_header
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html, dcc
import arrow
from omicron import tf

toolbar_style = {
    "display": "flex",
    "justify-content": "end",
    "align-items": "center",
    "height": "2rem",
    "width": "50vw"
}


def toolbar():
    end = arrow.now()
    start = end.shift(days=-365)

    label = html.P("策略名:", style={"margin-right": "1rem"})
    title = html.Div([label, html.P("网格交易", id="strategy-name")], style={"display": "flex", "flex-direction": "row"})
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
                    dbc.InputGroupText("-", style={"margin":"0 0.5rem"}),
                    dbc.Input(
                        id="end-date-input", type="date", value=end.format("YYYY-MM-DD")
                    ),
                ],
                size="sm",
                style={"width": "20rem"}
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
                ],size="sm",style={"width": "14rem"}
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

    return html.Div([title, toolbar], style={
        "display": "flex",
        "justify-content": "space-between",
    })


main = dbc.Container(
    toolbar(), fluid=True
)


@routing.dispatch("/backtest")
def init_page():
    return with_header(main)
