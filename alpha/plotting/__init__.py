import matplotlib.pyplot as plt
import numpy as np

from alpha.core.features import moving_average, predict_by_moving_average

cm = {5: "b", 10: "g", 20: "c", 60: "m", 120: "y", 250: "tab:orange", "raw": "tab:gray"}


def draw_trendline(
    close, ylen, ma_wins=None, canvas_size=60, desc: str = None, save_to: str = None
):
    """
    Draws moving average trendline on a graph.

    the trendline data is predicted by `predict_by_moving_average`
    """
    fig = plt.figure(figsize=(canvas_size // 10, canvas_size // 10), dpi=80)
    ax = fig.add_subplot(111)

    ma_wins = ma_wins or [5, 10, 20, 60]

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
    ax.plot(close[-n:], ".", color=cm["raw"], label="raw")
    for win in ma_wins:
        ypred, err = predict_by_moving_average(xclose, win, ylen, err_threshold=1)

        ma = moving_average(xclose, win)[-n:]
        # moving average line
        ax.plot(ma[:-ylen], color=cm[win], label=f"MA{win}")

        # 预测延伸线
        ax.plot(np.arange(n - ylen, n), ypred, color=cm[win], linestyle="--")
        ax.text(n, ypred[-1], f"{err:.3f}", color=cm[win])

    if save_to:
        fig.savefig(save_to)
