import numpy as np
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Candlestick:
    RED = '#FF4136'
    GREEN = '#3DAA70'
    TRANSPARENT = 'rgba(0,0,0,0)'
    LIGHT_GRAY = 'rgba(0, 0, 0, 0.1)'
    
    def __init__(self, bars: np.ndarray, ma_groups: List[int], show_volume: bool = True, title:str=None, **kwargs):
        self.title = title
        self.show_volume = show_volume
        self.bars = bars

        
        # traces for main area
        self.main_traces = {}
        
        # traces for indicator area
        self.ind_traces = {}
        
        self.ticks = [f"{x.month:02}-{x.day:02}" for x in bars["frame"]]
        
        # for every candlestick, it must contain a candlestick plot
        cs = go.Candlestick(x = self.ticks, 
                            open = bars["open"], 
                            high=bars["high"], 
                            low = bars["low"], 
                            close = bars["close"], 
                            line=dict({'width': 1}),
                            name = "K线",
                            **kwargs)

        # Set line and fill colors
        cs.increasing.fillcolor = 'rgba(255,255,255,0.9)'
        cs.increasing.line.color = self.RED
        cs.decreasing.fillcolor = self.GREEN
        cs.decreasing.line.color = self.GREEN
        
        self.main_traces["ohlc"] = cs
        
        if show_volume:
            self.add_indicator("volume")
            
        # 增加均线
        for win in ma_groups:
            name = f"ma{win}"
            ma = moving_average(bars["close"], win)
            n = len(ma)
            line = go.Scatter(y = ma, x = self.ticks[-n:], name=name, line=dict(width=1))
            self.main_traces[name] = line
            
    def add_indicator(self, indicator:str):
        if indicator == "volume":
            colors = np.repeat(self.RED, len(bars))
            colors[bars["close"] <= bars["open"]] = self.GREEN

            trace = go.Bar(x=self.ticks, y = self.bars["volume"], showlegend=False, marker={'color':colors})
            self.ind_traces[indicator] = trace
            
    def plot(self):
        rows = len(self.ind_traces) + 1
        
        cols = 1
        
        # todo: adjust row_heights
        fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, 
           vertical_spacing=0.03, subplot_titles=(self.title, *self.ind_traces.keys()), 
           row_width=[0.2, 0.7], specs=[[{"secondary_y": True}],[{"secondary_y": False}]])

        for name, trace in self.main_traces.items():
            fig.add_trace(trace, row=1, col=1)

        for i, (_, trace) in enumerate(self.ind_traces.items()):
            fig.add_trace(trace, row=i+2, col=1)
            
        fig.update(layout_xaxis_rangeslider_visible=False)
        fig.update_yaxes(showgrid=True, gridcolor=self.LIGHT_GRAY)
        fig.update_layout(plot_bgcolor=self.TRANSPARENT)
        fig.update_xaxes(type='category', tickangle=45, nticks=len(self.ticks)//3)
        
        fig.show()

def plot_candlestick(bars: np.ndarray, ma_groups: List[int], title:str=None, **kwargs):
    cs = Candlestick(bars, ma_groups, title=title)
    cs.plot()
