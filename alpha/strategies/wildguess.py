

from alpha.core.features import predict_by_moving_average
from types import FrameType
import numpy as np


class WildGuess(object):
    """
    A strategy that guesses randomly.
    """
    def __init__(self) -> None:
        self.ma_wins = [5, 10, 20, 30, 60]


    def scan(self, ):
        pass

    def get_pmae_err_threshold(self, win, frame_type:FrameType=FrameType.MIN30):
        if frame_type == FrameType.MIN30:
            return {
                5: 3e-3,
                10: 1e-3,
            }.get(win, 1e-4)
        elif frame_type == FrameType.DAY:
            return {
                5: 8e-3,
                10: 5e-3,
                20: 3e-3
            }.get(win, 1e-3)

    def _score_prediction(self, ypreds, threshold=1e-2):
        ymean = np.array(ypreds).flatten().mean()

        ypreds = np.array(ypreds)
        norm = ypreds/ymean

        results = [[]] * ypreds.shape[0]
        for i, x in enumerate(norm):
            for j, y in enumerate(norm):
                if i == j:
                    continue

                dist = np.mean(abs(x - y))

                if dist < threshold:
                    results[i].append(j)

        counts = [len(x) for x in results]
        idx = np.argmax(counts)
        return ypreds[idx], counts[idx]

    def guess(self, bars):
        close = bars['close']

        y_preds = []
        for win in self.ma_wins:
            _ypreds, _ = predict_by_moving_average(close, win, 5, self.get_pmae_err_threshold(win))

            if _ypreds is not None:
                y_preds.append(_ypreds)

        if len(y_preds) == 0:
            return None

        if len(y_preds) == 1:
            return y_preds[0][0]


