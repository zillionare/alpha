# noinspection PyUnresolvedReferences

import datetime
import os
from io import BytesIO
from typing import NewType, Optional, Union

import arrow

# Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from omicron.core.talib import moving_average
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.security import Security

Frame = NewType("Frame", (datetime.date, datetime.datetime, arrow.Arrow, str))


# Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running
# matplotlib.use("TkAgg")


class Candlestick:
    def __init__(
        self,
        frames: dict = None,
        fig_size=(12, 10),
        dpi=60,
        plot_window_size: int = 60,
        lfs=12,
        ma_lw=0.6,
    ):
        self.frames = frames or {
            "1d": [5, 10, 20, 60, 120, 250],
            "30m": [5, 10, 20, 60],
        }

        row = len(self.frames) * 2
        col = 1

        # how many n_bars will be drawn in the fig
        self.plot_window_size = plot_window_size

        self.dpi = dpi
        self.bw = (fig_size[0] * 4) / plot_window_size
        self.ma_lw = ma_lw
        self.lfs = lfs

        self.fig = plt.figure(figsize=fig_size, dpi=dpi)
        self.axes = []

        sub_height = 1 / len(self.frames) * 0.8
        sub_gap = 1 / len(self.frames) * 0.2

        for i in range(len(self.frames)):
            gs = GridSpec(
                2,
                1,
                hspace=0,
                top=1 - i * (sub_height + sub_gap),
                bottom=1 - (i + 1) * (sub_height),
                height_ratios=[3, 1],
            )
            self.axes.append(self.fig.add_subplot(gs[0, 0]))
            self.axes.append(self.fig.add_subplot(gs[1, 0]))

        self.cm = {
            5: "b",
            10: "g",
            20: "c",
            60: "m",
            120: "y",
            250: "tab:orange",
            "raw": "tab:gray",
        }

    def init_fig(self, y_lims=None):
        for i, ax in enumerate(self.axes):
            ax.cla()
            # if i%2 == 0:
            #     plt.setp(ax.get_xticklabels(), visible=False)
            #     ax.xaxis.set_tick_params(length=0)
            #     ax.grid(True, color="#cccccc", linestyle="-", linewidth="0.5")

            ax.yaxis.set_tick_params(labelsize=self.lfs)
            # ax.spines["top"].set_visible(False)
            # ax.spines["right"].set_visible(False)

            if y_lims is not None and i % 2 == 0:
                ax.set_ylim(y_lims[0])

            if y_lims is not None and i % 2 != 0:
                ax.set_ylim(y_lims[1])

        # self.axes[-1].spines["bottom"].set_visible(False)

        # self.fig.tight_layout(h_pad=0)
        # self.fig.canvas.draw()

    def close(self):
        self.fig.close()

    async def plot(self, code: str, end: Frame, title: str = None, save_to: str = None):
        # self.init_fig()

        sec = Security(code)
        end = arrow.get(end)

        plt.subplots_adjust(hspace=0)

        for i, ft in enumerate(self.frames.keys()):
            ma_wins = self.frames[ft]
            nbars = max(ma_wins) + self.plot_window_size
            frame_type = FrameType(ft)
            start = tf.shift(end, -nbars + 1, frame_type)

            bars = await sec.load_bars(start, end, frame_type)

            candlestick_ax = self.axes[2 * i]
            volume_ax = self.axes[2 * i + 1]

            self.plot_(bars, ma_wins, candlestick_ax, volume_ax)

        title = title or f"{code} {self.format_frames([end])[0]}"
        if title:
            self.fig.suptitle(title, fontsize=self.lfs)

        if save_to:
            file = os.path.join(save_to, f"{code}_{end.format('YY-MM-DD')}.png")
            self.fig.savefig(save_to, dpi=self.dpi)

    def plot_(
        self,
        bars: np.array,
        ma_groups,
        ax_candle_stick,
        ax_volume,
        title: str = None,
    ):
        """
        draw candlestick (with ma) and volume for given quotes

        args:
            bars: the dataframe which used to draw the candle stick
            ma_groups: hint for drawing which moving average line
            ax_candle_stick: the axis to draw the candlestick
            ax_volume: the axis to draw the volume n_bars
            title: the title of the fig
        return:
        """
        n = self.plot_window_size
        bars = bars.copy()
        factor = np.nanmax(bars["high"])
        for key in ["open", "close", "high", "low"]:
            bars[key] = bars[key] / factor

        bars["volume"] /= np.nanmean(bars["volume"])

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
        ax_volume.bar(range(n), volume, color=np.where(ups, "r", "g"), width=self.bw)

        positions = list(np.arange(n // 8 + 1) * 8)
        labels = [frames[i] for i in positions]
        ax_volume.set_xticks(positions)
        ax_volume.set_xticklabels(labels, rotation=45)

        if title:
            self.fig.suptitle(title, fontsize=self.lfs)

    def format_frames(self, frames):
        if hasattr(frames[0], "hour") and frames[0].hour != 0:
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
                lw=self.bw / 4,
                height=abs(o[i] - c[i]),
            )
            for i in range(len(bars))
        ]

        up_bottom = np.vstack(
            (
                np.arange(len(bars)),
                np.where(o >= c, o, c),
            )
        )
        up_top = np.vstack((np.arange(len(bars)), h))

        up = zip(np.transpose(up_top), np.transpose(up_bottom))

        down_top = np.vstack((np.arange(len(bars)), np.where(o >= c, c, o)))
        down_bottom = np.vstack((np.arange(len(bars)), l))

        down = zip(np.transpose(down_top), np.transpose(down_bottom))

        ax.add_collection(LineCollection(up, colors=edgecolor, linewidths=self.bw / 2))
        ax.add_collection(
            LineCollection(down, colors=edgecolor, linewidths=self.bw / 2)
        )

        rect_pc = PatchCollection(rects, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_collection(rect_pc)
        ax.autoscale_view()
