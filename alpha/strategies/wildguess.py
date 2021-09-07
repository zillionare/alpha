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
    replace_zero,
    top_n_argpos,
    volume_features,
)

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

    def volume_features(self, bars):
        """this version is not for machine learning

        Args:
            bars ([type]): [description]
        """
        win = 10

        close = fillna(bars["close"].copy())

        high = bars["high"].copy()
        low = bars["low"].copy()
        _open = bars["open"].copy()

        close = fillna(close.copy())
        high = fillna(high)
        low = fillna(low)
        _open = fillna(_open)

        avg = np.nanmean(bars["volume"][-win:])

        volume = replace_zero(bars["volume"].copy())

        # 涨跌
        flags = np.where((close > _open)[1:] & (close[1:] > close[:-1]), 1, -1)

        vr = volume[-win:] / avg
        indice = top_n_argpos(vr, 3)

        # 加上方向
        vr *= flags[-win:]

        # 按时间先后排列
        indice = np.sort(indice)
        return vr[indice]

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
