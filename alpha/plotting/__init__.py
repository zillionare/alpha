import arrow
import matplotlib.pyplot as plt
import numpy as np

from alpha.core.features import moving_average, predict_by_moving_average

cm = {5: "b", 10: "g", 20: "c", 60: "m", 120: "y", 250: "tab:orange", "raw": "tab:gray"}


def _format_frames(frames):
    if frames[0].hour != 0:
        fmt = "YY-MM-DD HH:mm"
    else:
        fmt = "YY-MM-DD"

    return [arrow.get(frame).format(fmt) for frame in frames]


def draw_trendline(
    bars, ylen, ma_wins=None, canvas_size=60, desc: str = None, save_to: str = None
):
    """
    Draws moving average trendline on a graph.

    the trendline data is predicted by `predict_by_moving_average`
    """
    fig = plt.figure(figsize=(canvas_size // 10, canvas_size // 10), dpi=80)
    ax = fig.add_subplot(111)

    ma_wins = ma_wins or [5, 10, 20, 60]

    close = bars["close"]
    if max(ma_wins) + canvas_size - 1 > len(close):
        raise ValueError("canvas_size is too big for the data")

    xclose = close[:-ylen]

    n = canvas_size

    if desc:
        ax.set_title(desc)

    # legend
    for i, win in enumerate(ma_wins):
        ax.text(0, 0.7 + 0.05 * i, f"ma{win}", color=cm[win], transform=ax.transAxes)
    # the raw data
    ax.plot(close[-n:], color=cm["raw"], label="raw")
    for win in ma_wins:
        ypred, err = predict_by_moving_average(xclose, win, ylen, err_threshold=1)

        ma = moving_average(xclose, win)[-n:]
        # moving average line
        ax.plot(
            np.arange(n - ylen - len(ma[:-ylen]), n - ylen), ma[:-ylen], color=cm[win]
        )
        ax.plot(np.arange(n - ylen, n), ma[-ylen:], color=cm[win])

        # 预测延伸线
        ax.plot(np.arange(n - ylen, n), ypred, color=cm[win], linestyle="--")
        ax.text(n, ypred[-1], f"{err:.3f}", color=cm[win])

    minc = min(close)
    maxc = max(close)
    splitter_y = np.arange(int(minc * 20), int(maxc * 20)) / 20
    splitter_x = [n - ylen] * len(splitter_y)
    ax.plot(splitter_x, splitter_y, "r:")

    frames = _format_frames(bars["frame"][-n:])
    if n // ylen * ylen == n:
        positions = list(np.arange(n // ylen) * ylen)
    else:
        positions = list(np.arange(n // ylen + 1) * ylen)

    labels = [frames[i] for i in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45)

    if save_to:
        fig.savefig(save_to)


def draw_ma_lines(bars, ma_wins=None, canvas_size=60, desc: str = None):
    ma_wins = ma_wins or [5, 10, 20, 60]
    nbars = max(ma_wins) + canvas_size
    if len(bars) < nbars:
        raise ValueError(f"not enough data, {nbars} required.")

    fig = plt.figure(figsize=(canvas_size // 10, canvas_size // 10), dpi=80)
    ax = fig.add_subplot(111)

    for win in ma_wins:
        ma = moving_average(bars["close"], win)[-canvas_size:]
        ax.plot(np.arange(canvas_size), ma, color=cm[win])
        ax.text(len(ma), ma[-1], f"ma{win}", color=cm[win])

    ax.text(0.5, 0.9, desc, transform=ax.transAxes)
