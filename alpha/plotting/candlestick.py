# noinspection PyUnresolvedReferences

from io import BytesIO
from typing import Optional, Tuple

# Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from omicron.core.talib import moving_average

# Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running
# matplotlib.use("TkAgg")


class Candlestick:
    def __init__(
        self,
        row=2,
        col=1,
        gridspec_kw=None,
        fig_size=(12, 8),
        dpi=60,
        plot_window_size: int = 60,
        lfs=12,
        ma_lw=0.6,
    ):
        # how many n_bars will be drawn in the fig
        self.plot_window_size = plot_window_size
        if gridspec_kw is None:
            gridspec_kw = {"height_ratios": [3, 1] * int(row / 2)}
        self.dpi = dpi
        self.bw = (fig_size[0] * 4)/ plot_window_size
        self.ma_lw = ma_lw
        self.lfs = lfs
        self.fig, self.axes = plt.subplots(
            row, col, gridspec_kw=gridspec_kw, figsize=fig_size, sharex=True, dpi=dpi
        )
        # todo: parameterize
        plt.subplots_adjust(hspace=0)
        self.cm = {5: "b", 10: "g", 20: "c", 60: "m", 120: "y", 250: "tab:orange", "raw": "tab:gray"}


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

        self.fig.tight_layout(h_pad=0)
        # self.fig.canvas.draw()

    def close(self):
        self.fig.close()

    def plot(
        self,
        bars: np.array,
        ma_groups,
        ax_candle_stick=None,
        ax_volume=None,
        name: str = None,
    ):
        """
        draw candlestick (with ma) and volume for given quotes

        args:
            bars: the dataframe which used to draw the candle stick
            ma_groups: hint for drawing which moving average line
            ax_candle_stick: the axis to draw the candlestick
            ax_volume: the axis to draw the volume n_bars
            name: for debug only
        return:
        """
        n = self.plot_window_size
        bars = bars.copy()
        factor = np.max(bars["high"])
        for key in ["open", "close", "high", "low"]:
            bars[key] = bars[key] / factor

        bars["volume"] /= np.nanmean(bars["volume"])

        ax_candle_stick = ax_candle_stick or self.axes[0]
        ax_volume = ax_volume or self.axes[1]

        ax_candle_stick.xaxis.set_visible(False)
        # draw candlestick
        # facecolor = self.calc_facecolor(bars[-n :])
        facecolor = "w"
        edgecolor = self.calc_edgecolor(bars[-n:])

        self.candle_stick_plot(
            ax_candle_stick,
            bars[-n:],
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

        # draw ma
        for i, win in enumerate(ma_groups or []):
            ma = moving_average(bars["close"], win)[-n:]
            line = Line2D(range(len(ma)), ma, color=self.cm[win], linewidth=self.ma_lw)
            ax_candle_stick.add_line(line)

        # draw volume
        ups = bars["close"][-n:] > bars["open"][-n:]

        volume = bars["volume"][-n:]
        frames = self.format_frames(bars["frame"][-n:])
        ax_volume.bar(
            range(n), volume, color=np.where(ups, "r", "g"), width=self.bw
        )

        positions = list(np.arange(n//8 + 1) * 8)
        labels = [frames[i] for i in positions]
        ax_volume.set_xticks(positions)
        ax_volume.set_xticklabels(labels, rotation=45)

    def format_frames(self, frames):
        if frames[0].hour != 0:
            fmt = "YY-MM-DD HH:mm"
        else:
            fmt = "YY-MM-DD"

        return [arrow.get(frame).format(fmt) for frame in frames]

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
        pct = (bars["close"][1:] + 0.01) / bars["close"][:-1] - 1

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
        rects = []
        o, c, h, l = bars["open"], bars["close"], bars["high"], bars["low"]
        rects = [
            Rectangle(
                (i - self.bw / 2.0, min(o[i], c[i])),
                width=self.bw,
                lw=self.bw/4,
                height=abs(o[i] - c[i]),
            )
            for i in range(len(bars))
        ]

        up_bottom = np.vstack((np.arange(len(bars)), np.where(o >= c, o, c), ))
        up_top = np.vstack((np.arange(len(bars)), h))

        up = zip(np.transpose(up_top), np.transpose(up_bottom))

        down_top = np.vstack((np.arange(len(bars)), np.where(o >= c, c, o)))
        down_bottom = np.vstack((np.arange(len(bars)), l))

        down = zip(np.transpose(down_top), np.transpose(down_bottom))

        ax.add_collection(LineCollection(up, colors=edgecolor, linewidths=self.bw/2))
        ax.add_collection(LineCollection(down, colors=edgecolor, linewidths=self.bw/2))

        rect_pc = PatchCollection(rects, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_collection(rect_pc)
        ax.autoscale_view()
