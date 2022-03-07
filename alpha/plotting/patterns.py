"""基于形态特征的绘图，比如压力线、支撑线等
"""
import matplotlib.pyplot as plt
import numpy as np
from omicron.talib.patterns import peaks_and_valleys, support_resist_lines


def plot_peaks_valleys(ts: np.ndarray, upthreshold=0.01, downthreshold=-0.01):
    pivots = peaks_and_valleys(ts, upthreshold, downthreshold)

    plt.plot(np.arange(len(ts)), ts, "k:", alpha=0.5)
    plt.plot(np.arange(len(ts))[pivots != 0], ts[pivots != 0], "k-")
    plt.scatter(np.arange(len(ts))[pivots == 1], ts[pivots == 1], color="g")
    plt.scatter(np.arange(len(ts))[pivots == -1], ts[pivots == -1], color="r")


def plot_support_resist_lines(ts: np.ndarray, upthreshold=0.01, downthreshold=-0.01):
    plot_peaks_valleys(ts)

    support, resist = support_resist_lines(ts)
    x = np.arange(len(ts) + 1)

    yresist = resist(x)
    ysupport = support(x)

    plt.plot(x, yresist, "g")
    plt.plot(x, ysupport, "r")
