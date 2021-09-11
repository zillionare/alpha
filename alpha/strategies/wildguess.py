import logging

import arrow
import numpy as np
import pandas as pd
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security

from alpha.core.features import (
    fillna,
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
        self.ma_wins = [5, 10, 20, 30, 60, 120]
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

                pred_profit, ypred, pred_credits, rsi, vf, rh = result

                if pred_profit < profit:
                    logger.debug("%s is predictable, but potential profit is low")
                    continue

                vec = [sec.display_name, code, pred_profit, ypred, pred_credits]
                vec.extend(rsi)
                vec.extend(vf)
                vec.extend(rh)
                results.append(vec)
                print(f"{code} {pred_profit:.0%}")
            except Exception as e:
                continue

        df = pd.DataFrame(
            results,
            columns=[
                "name",
                "code",
                "pred_profit",
                "ypred",
                "pred_credits",
                "rsi_1",
                "rsi_2",
                "rsi_3",
                "vf_1",
                "vf_2",
                "vf_3",
                "rh",
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

        pred_credits = 0
        ypred = 0

        for win in self.ma_wins:
            _ypreds, _ = predict_by_moving_average(
                close, win, 5, self.get_pmae_err_threshold(win, frame_type)
            )

            if _ypreds is None:
                logger.debug("can not predict trend of %s", code)
                continue

            # 如果长线看空，则不操作
            if _ypreds[-1] < close[-1] and win in [20, 30, 60, 120]:
                logger.debug("%s is not going up", code)
                return None

            pred_credits += 1
            ypred = max(ypred, max(_ypreds))

        rsi = relative_strength_index(close)
        if np.any(rsi[-3:] > 90):
            logger.debug("RSI of %s at %s is too high", code, bars["frame"][-5:])
            return None

        vf = self.volume_features(bars)

        c0 = bars["close"][-1]
        pred_profit = ypred / c0 - 1

        rh = []
        rh.append(relation_with_prev_high(close, len(close))[0])

        return pred_profit, ypred, pred_credits, rsi[-3:], vf, rh

    def clams(self, bars, code: str, adv: float, frame_type: FrameType = FrameType.DAY):
        """涨幅3~5%，两连阳，均线5,10,20预测向上，放量，连续上涨不超过adv%"""
        close = bars["close"]
        vf = top_volume_direction(bars)

        if vf[-1] < 2:
            logger.debug("volume of %s is not increased", code)
            return None

        pcr = close[1:] / close[:-1]
        if np.any(pcr[-2:] < 1.01) or not (1.025 <= pcr[-1] <= 1.06):
            logger.debug("%s is not continuousely increasing", code)
            return None

        for win in [5, 10, 20]:
            _ypreds, _ = predict_by_moving_average(
                close, win, 5, self.get_pmae_err_threshold(win, frame_type)
            )

            if _ypreds is None:
                logger.debug("can not predict trend of %s", code)
                continue

            if _ypreds[-1] < close[-1] and win in [20, 30, 60, 120]:
                logger.debug("%s is not going up", code)
                return None
