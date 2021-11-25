# noinspection PyUnresolvedReferences

import datetime
import os
from io import BytesIO
from typing import List, NewType, Optional, Union

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
from matplotlib.gridspec import GridSpec
import logging

Frame = NewType("Frame", (datetime.date, datetime.datetime, arrow.Arrow, str))


# Cannot load backend 'TkAgg' which requires the 'tk' interactive framework, as 'headless' is currently running
# matplotlib.use("TkAgg")

logger = logging.getLogger(__name__)


plt.rc(
    "font",
    family=[
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "Heiti TC",
        "Songti SC",
        "STHeitiSC-Light",
    ],
)


class Candlestick:
    def __init__(
        self,
        frames: dict = None,
        dpi=60,
        n_plot_bars: int = 60,
        bw=20,
        lw=20,
        font_size=12,
        hw_ratio=0.6,
    ):
        """[summary]

        Args:
            frames (dict, optional): which level bars to draw. Defaults to "1d" and "30m".
            dpi (int, optional): [description]. Defaults to 60.
            n_plot_bars (int, optional): how many bars will be displayed on the canvas. Defaults to 60.
            font_size (int, optional): label font size. Defaults to 16.
            bw (int, optional): bar width in pixels, Defaults to 20.
            lw (int, optional): line width in pixels, Defaults to 20,
            hw_ratio (float, optional): the aspect ratio between height/width, Defaults to 0.75.
        """
        self.frames = frames or {
            "1d": [5, 10, 20, 60, 120, 250],
            "30m": [5, 10, 20, 60],
        }

        plt.rc("font", size=font_size)
        plt.rcParams["axes.unicode_minus"] = False

        # how many n_bars will be drawn in the fig
        self.plot_window_size = n_plot_bars

        self.dpi = dpi

        self.fig_size = (
            n_plot_bars * bw / (2 * dpi),
            n_plot_bars * bw * hw_ratio / (2 * dpi),
        )

        # bar width in inches
        self.bw = 1.8 * bw / dpi
        # line width of moving average line
        self.lw = 1.8 * lw / dpi

        self.fig = plt.figure(figsize=self.fig_size, dpi=dpi)
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
            20: "r",
            30: "c",
            60: "m",
            120: "y",
            250: "k",
            "raw": "tab:gray",
        }

    def reset(self, y_lims=None):
        """reset so we can draw another one"""
        for i, ax in enumerate(self.axes):
            ax.cla()

    def close(self):
        self.fig.close()

    async def plot(
        self,
        code: str,
        end: Frame,
        title: str = None,
        save_to: str = None,
    ):
        # self.init_fig()

        sec = Security(code)
        end = arrow.get(end)

        plt.subplots_adjust(hspace=0)

        for i, ft in enumerate(self.frames.keys()):
            ma_wins = self.frames[ft]
            nbars = max(ma_wins) + self.plot_window_size
            frame_type = FrameType(ft)
            start = tf.shift(tf.floor(end, frame_type), -nbars + 1, frame_type)

            try:
                bars = await sec.load_bars(start, end, frame_type)
            except Exception as e:
                logger.exception(e)
                return

            candlestick_ax = self.axes[2 * i]
            volume_ax = self.axes[2 * i + 1]

            self.plot_(bars, ma_wins, candlestick_ax, volume_ax)

        title = title or f"{code} {self.format_frame(end)}"
        if title:
            self.fig.suptitle(title)

        if save_to:
            file = os.path.join(save_to, f"{code}_{end.format('YY-MM-DD')}.png")
            self.fig.savefig(file, dpi=self.dpi)

    def plot_bars(
        self, bars: np.array, title: str = None, save_as: str = None, signals=[]
    ):
        """给定一个bar数组，绘制k线图

        要求在构造CandleStick对象时，指定一个与此对应的惟一的frame设置。

        Args:
            bars (np.array): [description]
            title (str, optional): [description]. Defaults to None.
            save_as (str, optional): [description]. Defaults to None.
            signals (list, optional): [(pos, marker, color)]
        """
        assert len(self.frames) == 1

        ma_groups = list(self.frames.values())[0]
        self.plot_(bars, ma_groups, self.axes[0], self.axes[1], title, signals)

        if save_as:
            self.fig.savefig(save_as, dpi=self.dpi)

    def plot_(
        self,
        bars: np.array,
        ma_groups,
        ax_candle_stick,
        ax_volume,
        title: str = None,
        signals: List = [],
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
        n = min(self.plot_window_size, len(bars))
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
            signals=signals,
        )

        # draw ma
        for i, win in enumerate(ma_groups or []):
            ma = moving_average(bars["close"], win)[-n:]
            line = Line2D(range(len(ma)), ma, color=self.cm[win], linewidth=self.lw)
            ax_candle_stick.add_line(line)

        # draw volume
        ups = bars["close"][-n:] > bars["open"][-n:]

        volume = bars["volume"][-n:]
        ax_volume.bar(range(n), volume, color=np.where(ups, "r", "g"), width=self.bw)

        labels = self.format_labels(bars["frame"][-n:])
        label_pos = list(np.arange(n))

        ax_volume.set_xticks(label_pos)
        ax_volume.set_xticklabels(labels, rotation=45)

        if title:
            self.fig.suptitle(title)

    def format_labels(self, frames):
        formatted = []
        gap = 4 if hasattr(frames[0], "hour") else 2
        for i, frame in enumerate(frames):
            if i % gap == 0:
                formatted.append(self.format_frame(frame))
            else:
                formatted.append("")

        return formatted

    def format_frame(self, frame):
        if hasattr(frame, "hour") and frame.hour != 0:
            fmt = "MM-DD HH:mm"
        else:
            fmt = "YY-MM-DD"

        return arrow.get(frame).format(fmt)

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
        self,
        ax,
        bars: np.array,
        facecolor: np.array = None,
        edgecolor: np.array = None,
        signals=[],
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
                width=self.bw * 0.94,  # leave some space between bars
                lw=self.lw,
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

        # the line width sounds too wide, but it's the only normal way to draw the line
        ax.add_collection(LineCollection(up, colors=edgecolor, linewidths=self.lw))
        ax.add_collection(LineCollection(down, colors=edgecolor, linewidths=self.lw))

        rect_pc = PatchCollection(rects, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_collection(rect_pc)

        for pos, text, color in signals:
            ax.text(pos - 0.5, c[pos], text, color=color)
        ax.autoscale_view()
