from alpha.core.commons import plateaus
import numpy as np
from typing import List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import arrow
import talib
from omicron import moving_average, peaks_and_valleys
from coretypes import Frame
from collections import defaultdict


class Candlestick:
    RED = "#FF4136"
    GREEN = "#3DAA70"
    TRANSPARENT = "rgba(0,0,0,0)"
    LIGHT_GRAY = "rgba(0, 0, 0, 0.1)"

    def __init__(
        self,
        bars: np.ndarray,
        ma_groups: List[int] = [5, 10, 20, 60],
        title: str = None,
        show_volume=True,
        show_peaks=False,
        **kwargs,
    ):
        self.title = title
        self.bars = bars

        # traces for main area
        self.main_traces = {}

        # traces for indicator area
        self.ind_traces = {}

        self.ticks = np.array([self._format_tick(x) for x in bars["frame"]])

        # for every candlestick, it must contain a candlestick plot
        cs = go.Candlestick(
            x=self.ticks,
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            line=dict({"width": 1}),
            name="K线",
            **kwargs,
        )

        # Set line and fill colors
        cs.increasing.fillcolor = "rgba(255,255,255,0.9)"
        cs.increasing.line.color = self.RED
        cs.decreasing.fillcolor = self.GREEN
        cs.decreasing.line.color = self.GREEN

        self.main_traces["ohlc"] = cs

        if show_volume:
            self.add_indicator("volume")

        if show_peaks:
            print("add peaks")
            self.add_main_trace("peaks")

        # 增加均线
        for win in ma_groups:
            name = f"ma{win}"
            ma = moving_average(bars["close"], win)
            n = len(ma)
            line = go.Scatter(y=ma, x=self.ticks[-n:], name=name, line=dict(width=1))
            self.main_traces[name] = line

    def _format_tick(self, tm: Frame) -> str:
        return f"{tm.year:02}-{tm.month:02}-{tm.day:02}"

    def add_main_trace(self, trace_name: str, **kwargs):
        """add trace to main plot"""
        if trace_name == "peaks":
            self.mark_peaks_and_valleys(
                kwargs.get("up_thres", 0.03), kwargs.get("down_thres", -0.03)
            )

        # 标注矩形框
        if trace_name == "bbox":
            self.mark_bounding_box(kwargs.get("boxes"))

        # 回测结果
        if trace_name == "bt":
            self.mark_backtest_result(kwargs.get("bt"))

    def mark_backtest_result(self, result: dict):
        """标记买卖点和回测数据

        Args:
            result: 回测结果，包含trades和每日assets,其中：
                - trades: 交易记录列表，每个元素是一个dict，包含以下字段：
                    - time: 交易日期
                    - price: 交易价格
                    - volume: 交易数量
                    - order_side: 交易方向，买入或卖出
                    - security: 股票代码
                - assets: 每日资产，字典，key为日期，value为资产
        """
        trades = result.get("trades")
        assets = result.get("assets")

        x, y, labels = [], [], []
        hover = []
        labels_color = defaultdict(list)

        for trade in trades:
            trade_date = arrow.get(trade["time"]).date()
            asset = assets.get(trade_date)

            security = trade["security"]
            price = trade["price"]
            volume = trade["volume"]

            side = trade["order_side"]

            x.append(self._format_tick(trade_date))

            bar = self.bars[self.bars["frame"] == trade_date]
            if side == "买入":
                hover.append(
                    f"总资产:{asset}<br><br>{side}:{security}<br>买入价:{price}<br>股数:{volume}"
                )

                y.append(bar["high"][0] * 1.1)
                labels.append("B")
                labels_color["color"].append(self.RED)

            else:
                y.append(bar["low"][0] * 0.99)

                hover.append(
                    f"总资产:{asset}<hr><br>{side}:{security}<br>卖出价:{price}<br>股数:{volume}"
                )

                labels.append("S")
                labels_color["color"].append(self.GREEN)

        trace = go.Scatter(
            x=x,
            y=y,
            mode="text",
            text=labels,
            name="backtest",
            hovertext=hover,
            textfont=labels_color,
        )

        self.main_traces["bs"] = trace

    def mark_peaks_and_valleys(self, up_thres: float = 0.03, down_thres: float = -0.03):
        bars = self.bars

        flags = peaks_and_valleys(
            bars["close"].astype(np.float64), up_thres, down_thres
        )
        ticks_up = self.ticks[:-1][flags == 1]
        y_up = bars["high"][flags == 1] * 1.005

        ticks_down = self.ticks[:-1][flags == -1]
        y_down = bars["low"][flags == -1] * 0.995

        trace = go.Scatter(
            mode="markers",
            x=ticks_up,
            y=y_up,
            marker_symbol="triangle-down",
            name="peak",
        )
        self.main_traces["peaks"] = trace

        trace = go.Scatter(
            mode="markers",
            x=ticks_down,
            y=y_down,
            marker_symbol="triangle-up",
            name="valley",
        )
        self.main_traces["valleys"] = trace

    def mark_bounding_box(self, boxes: List[Tuple]):
        """bbox是标记在k线图上某个区间内的矩形框，它以该区间最高价和最低价为上下边。

        Args:
            boxes : 各个bbox的起点和宽度。
        """
        x, y = [], []
        for box in boxes:
            i, width = box
            if len(x):
                x.append(None)
                y.append(None)

            h = max(self.bars["high"][i : i + width])
            l = min(self.bars["low"][i : i + width])
            x.extend(self.ticks[[i, i + width, i + width, i, i]])
            y.extend((h, h, l, l, h))

            trace = go.Scatter(x=x, y=y, fill="toself", name="bbox")
            self.main_traces["bbox"] = trace

    def add_indicator(self, indicator: str):
        """"向k线图中增加技术指标"""
        if indicator == "volume":
            colors = np.repeat(self.RED, len(self.bars))
            colors[self.bars["close"] <= self.bars["open"]] = self.GREEN

            trace = go.Bar(
                x=self.ticks,
                y=self.bars["volume"],
                showlegend=False,
                marker={"color": colors},
            )
        elif indicator == "rsi":
            rsi = talib.RSI(self.bars["close"].astype(np.float64))
            trace = go.Scatter(x=self.ticks, y=rsi, showlegend=False)
        else:
            raise ValueError(f"{indicator} not supported")

        self.ind_traces[indicator] = trace

    def plot(self):
        rows = len(self.ind_traces) + 1
        specs = [[{"secondary_y": False}]] * rows
        specs[0][0]["secondary_y"] = True

        row_heights = [0.7, *([0.2] * (rows - 1))]
        cols = 1

        fig = make_subplots(
            rows=rows,
            cols=cols,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(self.title, *self.ind_traces.keys()),
            row_heights=row_heights,
            specs=specs,
        )

        for _, trace in self.main_traces.items():
            fig.add_trace(trace, row=1, col=1)

        for i, (_, trace) in enumerate(self.ind_traces.items()):
            fig.add_trace(trace, row=i + 2, col=1)

        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_yaxes(showgrid=True, gridcolor=self.LIGHT_GRAY)
        fig.update_layout(plot_bgcolor=self.TRANSPARENT)
        fig.update_xaxes(type="category", tickangle=45, nticks=len(self.ticks) // 3)

        fig.show()


def plot_candlestick(
    bars: np.ndarray, ma_groups: List[int], title: str = None, **kwargs
):
    cs = Candlestick(bars, ma_groups, title=title)
    cs.plot()
