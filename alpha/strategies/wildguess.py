import logging

import arrow
import numpy as np
import pandas as pd
from omicron.models.timeframe import TimeFrame as tf
from coretypes import FrameType
from omicron.models.securities import Securities
from omicron.models.stock import Stock

from alpha.core.features import (
    moving_average,
    predict_by_moving_average,
    relation_with_prev_high,
    relative_strength_index,
)
from alpha.features.volume import top_volume_direction

logger = logging.getLogger(__name__)


class WildGuess(object):
    """
    A strategy that guesses randomly.
    """

    def __init__(self) -> None:
        self.ma_wins = [5, 10, 20, 30, 60]
        self.nbars = max(self.ma_wins) + 20

    async def scan(self, frame_type: str, codes: list = None, profit=0.15, end=None):
        results = []
        frame_type = FrameType(frame_type)

        if end is None:
            if frame_type == FrameType.DAY:
                end = arrow.now().date()
            else:
                end = tf.floor(arrow.now(), frame_type)

        codes = codes or Securities().choose(["stock"])
        for code in codes:
            try:
                sec = Security(code)

                start = tf.shift(end, -self.nbars + 1, frame_type)

                bars = await sec.load_bars(start, end, frame_type)
                if (
                    bars is None
                    or len(bars) != self.nbars
                    or np.count_nonzero(np.isnan(bars["close"])) > len(bars) * 0.1
                ):
                    continue

                try:
                    result = self.guess(bars, code, frame_type)
                except Exception:
                    continue

                if result is None:
                    continue

                pred_profit, up_trends, rsi, vf, rh = result

                if pred_profit < profit:
                    continue

                print(f"{code} {pred_profit:.0%}")
                vec = [sec.display_name, code, pred_profit, up_trends, rh]
                vec.extend(rsi)
                vec.extend(vf)
                results.append(vec)
            except Exception as e:
                continue

        df = pd.DataFrame(
            results,
            columns=[
                "name",
                "code",
                "pred_profit",
                "up_trends",
                "rh",
                "rsi_1",
                "rsi_2",
                "rsi_3",
                "vf_1",
                "vf_2",
                "vf_3",
            ],
        )
        return df

    def get_pmae_err_threshold(self, win, frame_type: FrameType = FrameType.MIN30):
        if frame_type == FrameType.MIN30:
            return {
                5: 3e-3,
                10: 1e-3,
            }.get(win, 1e-4)
        elif frame_type == FrameType.DAY:
            return {5: 8e-3, 10: 5e-3, 20: 3e-3}.get(win, 3e-3)

    def guess(self, bars, code: str, frame_type: FrameType):
        close = bars["close"]

        up_trends = 0

        ypreds, _ = predict_by_moving_average(
            close, 5, 5, self.get_pmae_err_threshold(5, frame_type)
        )
        if ypreds is None or ypreds[0] < close[-1]:
            return None

        pred_profit = ypreds[-1] / close[-1] - 1

        for win in self.ma_wins[1:]:
            _ypreds, _ = predict_by_moving_average(
                close, win, 5, self.get_pmae_err_threshold(win, frame_type)
            )

            if _ypreds is None:
                continue

            ma = moving_average(close, win)
            if _ypreds[-1] <= ma[-1]:  # 均线是下降的
                return None

            up_trends += 1

        rsi = relative_strength_index(close, 6)
        if np.any(rsi[-3:] > 97):
            return None

        vf = top_volume_direction(bars, n=16)

        rh = relation_with_prev_high(close, len(close))[0]

        return pred_profit, up_trends, rsi[-3:], vf, rh
