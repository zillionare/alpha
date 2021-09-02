# noinspection PyUnresolvedReferences
import logging
from io import BytesIO
from typing import Optional, Tuple

import matplotlib

# Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# noinspection PyUnresolvedReferences
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


class Candlestick:
    def __init__(
        self,
        row=6,
        col=1,
        gridspec_kw=None,
        fig_size=(8, 8),
        dpi=60,
        plot_window_size: int = 60,
        lw=0.4,
        bw=0.6,
        lfs=12,
        ma_lw=0.6,
    ):
        # how many n_bars will be drawn in the fig
        self.plot_window_size = plot_window_size
        if gridspec_kw is None:
            gridspec_kw = {"height_ratios": [3, 1] * int(row / 2)}
        self.dpi = dpi
        self.lw = lw
        self.bw = bw
        self.ma_lw = ma_lw
        self.lfs = lfs
        self.fig, self.axes = plt.subplots(
            row, col, gridspec_kw=gridspec_kw, figsize=fig_size, sharex=True, dpi=dpi
        )
        # todo: parameterize
        plt.subplots_adjust(hspace=0.02)

    def init_fig(self, y_lims=None):
        for i, ax in enumerate(self.axes):
            ax.cla()
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.xaxis.set_tick_params(length=0)
            ax.yaxis.set_tick_params(labelsize=self.lfs)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if i % 2 == 0:
                ax.grid(True, color="#cccccc", linestyle="-", linewidth="0.5")

            if y_lims is not None and i % 2 == 0:
                ax.set_ylim(y_lims[0])

            if y_lims is not None and i % 2 != 0:
                ax.set_ylim(y_lims[1])

        self.axes[-1].spines["bottom"].set_visible(False)

        self.fig.tight_layout(h_pad=0.02)
        # cls.fig.canvas.draw()
        pass

    def close(self):
        self.fig.close()

    def plot(
        self,
        bars: np.array,
        ma_groups,
        ax_candle_stick=None,
        ax_turnover=None,
        name: str = None,
    ):
        """
        draw candlestick (with ma) and volume for given quotes

        args:
            bars: the dataframe which used to draw the candle stick
            ma_groups: hint for drawing which moving average line
            ax_candle_stick: the axis to draw the candlestick
            ax_turnover: the axis to draw the volume n_bars
            name: for debug only
        return:
        """
        # hide tick
        ax_candle_stick = ax_candle_stick or self.axes[0]
        ax_turnover = ax_turnover or self.axes[1]

        # draw candlestick
        facecolor = self.calc_facecolor(bars[-1 * self.plot_window_size :])
        edgecolor = self.calc_edgecolor(bars[-1 * self.plot_window_size :])

        self.candle_stick_plot(
            ax_candle_stick,
            bars[-1 * self.plot_window_size :],
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

        # draw ma
        for i, win in enumerate(ma_groups or []):
            ma = (
                bars["close"]
                .interpolate()
                .rolling(win)
                .mean()[-1 * self.plot_window_size :]
            )
            ma.index -= ma.index[0]
            line = Line2D(ma.index, ma.values, color=f"C{i}", linewidth=self.ma_lw)
            ax_candle_stick.add_line(line)

        # draw turnover
        ups = (
            bars["close"][-self.plot_window_size :]
            > bars["open"][-self.plot_window_size :]
        )
        turnover = bars["turnover"][-self.plot_window_size :]
        ax_turnover.bar(
            range(len(turnover)), turnover, color=np.where(ups, "r", "g"), width=self.bw
        )

    def calc_edgecolor(self, bars: np.array) -> np.array:
        """
        if close >= open, then draw the box in green line; else in red line
        """
        colors = np.array(["r"] * len(bars))
        colors[bars["close"] < bars["open"]] = "g"
        return colors

    def calc_facecolor(self, bars: np.array) -> np.array:
        """
        if close >= open and reach advance limit, draw solid box in red;
        if advance limit is not reached, draw empty box in red;
        else draw solid green box

        Args:
            bars: the bars data which used to draw the candle stick
        """
        # fixme: use real buy/sell limit function
        pct = (bars["close"] + 0.01) / bars["close"].shift(1) - 1

        colors = np.array(["w"] * len(bars))
        colors[np.where(pct <= -0.099)] = "g"
        colors[np.where(pct >= 0.099)] = "r"

        return colors

    def candle_stick_plot(
        self, ax, bars: np.array, facecolor: np.array = None, edgecolor: np.array = None
    ):
        """
        args:
            facecolor: facecolor (fill the box body) for each box)
            edgecolor: line color
            bars: normalized df
            w: the width of stem part
        """
        vertex_top = np.zeros((len(bars), 2, 2))
        vertex_bottom = np.zeros((len(bars), 2, 2))

        rects = []
        for i in range(len(bars)):
            o, c, l, h = (
                bars.iloc[i]["open"],
                bars.iloc[i]["close"],
                bars.iloc[i]["low"],
                bars.iloc[i]["high"],
            )
            if o >= c:
                vertex_top[i] = [[i, o], [i, h]]
                vertex_bottom[i] = [[i, l], [i, c]]
            else:
                vertex_top[i] = [[i, c], [i, h]]
                vertex_bottom[i] = [[i, l], [i, o]]

            rect = Rectangle(
                (i - self.bw / 2.0, min(bars.iloc[i]["open"], bars.iloc[i]["close"])),
                width=self.bw,
                lw=self.lw,
                height=abs(bars.iloc[i]["open"] - bars.iloc[i]["close"]),
            )
            rects.append(rect)

        for vertex in [vertex_top, vertex_bottom]:
            line_collection = LineCollection(vertex, color=edgecolor)
            ax.add_collection(line_collection)

        rect_pc = PatchCollection(rects, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_collection(rect_pc)
