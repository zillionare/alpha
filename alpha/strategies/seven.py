import logging
import os

import arrow
import cfg4py
import numpy as np
from omicron.models.timeframe import TimeFrame as tf
from coretypes import FrameType
from omicron.models.stock import Stock
from sklearn.metrics import mean_absolute_error

from alpha.core.features import (
    fillna,
    moving_average,
    polyfit,
    predict_by_moving_average,
    relative_strength_index,
    top_n_argpos,
    transform_to_change_pct,
    transform_y_by_change_pct,
    volume_features,
)
from alpha.strategies.base_xgboost_strategy import BaseXGBoostStrategy
from alpha.utils.data import DataBunch, make_dataset

cfg = cfg4py.init()

logger = logging.getLogger(__name__)


class Seven(BaseXGBoostStrategy):
    def __init__(self, *args, **kwargs):
        home = os.path.expanduser(cfg.alpha.data_home)
        super().__init__("Seven", home)

        self.wins = [5, 10, 20, 30, 60]

        self.n_xbars = max(self.wins) * 2
        self.n_ybars = 5
        self.bins = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2]
        self.nbuckets = len(self.bins) + 2

        # volume feature window
        self.vol_win = max(self.wins)

    def x_transform(self, xbars):
        vec = []
        xclose = fillna(xbars["close"].copy())

        mas = []
        # add moving average trend line
        for win in self.wins:
            fit_win = max(7, win)
            ma = moving_average(xclose, win)[-fit_win:]
            ma /= ma[0]
            (a, b, c), pmae = polyfit(ma)
            # c should be always 1
            vec.extend([a, b, pmae])

            mas.append(ma[-1])

        # add volume features
        vec.extend(volume_features(xbars, self.vol_win))

        # add last 3 days rsi features
        rsi = relative_strength_index(xclose)[-3:]
        vec.extend(rsi / 100)

        # 最短窗口到最长窗口的ma之前的偏转度，在[-1,1]之间
        vec.append((mas[0] - mas[-1]) / (mas[0] + mas[-1]))

        return vec

    def y_transform(self, ybars, xbars, code):
        xclose = fillna(xbars["close"].copy())
        yclose = ybars["close"]

        yn = len(yclose)

        c0 = xclose[-1]

        if abs(min(yclose) / c0 - 1) < abs(max(yclose) / c0 - 1):
            agg = max
        else:
            agg = min

        y = agg(yclose) / c0
        for win in self.wins:
            fit_win = max(7 + win, 2 * win)
            pred_close, pmae = predict_by_moving_average(
                xclose[-fit_win:], win, yn, err_threshold=1
            )

            # use dynamic threshold, bigger threshold for large fluctuation
            if self.y_to_bucket(y) in [0, self.nbuckets - 1, self.nbuckets]:
                err_threshold = 0.02
            else:
                err_threshold = 0.01

            err = mean_absolute_error(yclose, pred_close) / yclose.mean()
            # means price can be predicted by moving average
            if err < err_threshold and pmae < err_threshold:
                return y, self.y_to_bucket(y)

        if y > 1.2 or y < 0.85:
            xend = xbars["frame"][-1]
            logger.info(
                "adv/dec rate of %s in 5 days since %s reach at %s, while cannot be predicted by ma",
                code,
                xend,
                y,
            )

        return 0, self.nbuckets - 1

    def y_to_bucket(self, y):
        if y == 0:
            return self.nbuckets - 1

        # [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2]
        for i, b in enumerate(self.bins):
            if y - 1 < b:
                return i
        else:
            return len(self.bins) + 1

    async def make_dataset(
        self, total: int, notes: str = None, version=None
    ) -> DataBunch:
        transformers = {
            FrameType.DAY: {
                "func": self.x_transform,
                "bars_len": self.n_ybars + self.n_xbars,
            }
        }
        target_transformer = self.y_transform
        target_win = self.n_ybars

        labels = " ".join([f"{k}->{v}" for k, v in enumerate(self.bins)])
        notes = notes or f"name: {self.name}, labels: {labels}"
        return await make_dataset(
            transformers,
            target_transformer,
            target_win,
            total,
            nbuckets=self.nbuckets,
            notes=notes,
            epoch=200,
        )

    async def check(
        self,
        code: str,
        xend: str,
        n: int,
        ft: FrameType = FrameType.DAY,
        wins=None,
        disp_nbars=40,
    ):
        """check specific stock's status

        Args:
            code (str): [description]
            end (str): [description]
            n (int): [description]
            ft (FrameType, optional): [description]. Defaults to FrameType.DAY.
        """
        import matplotlib.pyplot as plt

        color_map = {
            5: "b",
            10: "c",
            20: "k",
            30: "g",
            60: "m",
            120: "r",
            250: "y",
        }

        sec = Security(code)
        xend = tf.shift(arrow.get(xend), 0, ft)
        start = tf.shift(xend, -n, ft)
        yend = tf.shift(xend, 5, ft)
        bars = await sec.load_bars(start, yend, ft)

        close = bars["close"]
        xclose = close[:-5]
        yclose = close[-5:]

        plt.plot(close[-disp_nbars:], "--", color="tab:red")

        plt.text(0.1, 0.9, f"{code} {xend}", transform=plt.gca().transAxes)

        wins = wins or self.wins
        for i, win in enumerate(wins):
            fit_win = max(7 + win, 2 * win)
            ypred, pmae_ma = predict_by_moving_average(
                xclose[-fit_win:], win, 5, err_threshold=1
            )

            pmae_y_ypred = mean_absolute_error(yclose, ypred) / yclose.mean()

            ma = moving_average(xclose, win)[-disp_nbars + 5 :]
            plt.plot(ma, color=color_map[win])
            plt.plot(
                np.arange(disp_nbars - 5, disp_nbars), ypred, ".", color=color_map[win]
            )

            print(f"{win}日: 均线误差 {pmae_ma:.3f} 预测误差 {pmae_y_ypred:.2f} 预测：{ypred}")
